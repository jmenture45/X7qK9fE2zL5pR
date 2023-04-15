#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <nlopt.h>
#include "DRV2700.h"
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <time.h>
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8
#define GRADIENT_STEP_SIZE 1e-5
#define MAX_ITERATIONS 1000


// State structure containing glucose and insulin levels
typedef struct {
    double glucose;
    double insulin;
} State;

// Hovorka model parameters structure
typedef struct {
    double k1, k2, k3;
    double k4, k5, k6;
    double u_basal, u_meal;
} HovorkaParams;

// Model Predictive Control parameters structure
typedef struct {
    double glucose_target; //this was originally glucose_setpoint so let's make sure the naming is consistent throughout the code
    double glucose_min;
    double insulin_min;
    double insulin_max;
    int prediction_horizon;
    double cost_weight;
    double insulin_rate_max;
    double insulin_rate_min;
} MPCParams;

// Model Predictive Control parameters structure Option 1
typedef struct {
    double (*find_optimal_insulin_fn)(State *state, void *model_params, MPCParams *mpc_params, int prediction_horizon);
} OptimizationAlgorithm;

// Model Predictive Control parameters structure Option 2
typedef struct {
    double (*find_optimal_insulin_fn)(State *, HovorkaParams *, MPCParams *, int);
} OptimizationAlgorithm;

// Perform one step of the Hovorka model
void hovorka_step(State *state, HovorkaParams *params, double u, double dt) {
    double glucose_new = state->glucose - params->k1 * state->glucose * dt + params->k2 * state->insulin * dt;
    double insulin_new = state->insulin + params->k3 * u * dt - params->k4 * state->insulin * dt;
    state->glucose = glucose_new;
    state->insulin = insulin_new;
}
// Initialize Hovorka model parameters based on body weight
HovorkaParams hovorka_parameters(double BW) {
    HovorkaParams params;

    // Initialize parameters with extracted values and formulas from the provided Python script
    double V_I = 0.12 * BW;
    double V_G = 0.16 * BW;
    double F_01 = 0.0097 * BW;
    double EGP_0 = 0.0161 * BW;

    double S_IT = 51.2e-4;
    double S_ID = 8.2e-4;
    double S_IE = 520e-4;

    double k_a1 = 0.006;
    double k_b1 = S_IT * k_a1;
    double k_a2 = 0.06;
    double k_b2 = S_ID * k_a2;
    double k_a3 = 0.03;
    double k_b3 = S_IE * k_a3;

    double k_12 = 0.066;
    double k_e = 0.138;

    params.k1 = F_01 / V_G + k_12;
    params.k2 = k_b1 * k_a1 / V_I;
    params.k3 = k_b2 * k_a2 / V_I;
    params.k4 = k_12;
    params.k5 = k_b1 * k_a1 / V_G;
    params.k6 = EGP_0 * k_b3 * k_a3 / V_G;
    params.u_basal = (F_01 + EGP_0) / S_IT;
    params.u_meal = 0.0; // Set to 0.0 initially; can be updated if needed

    return params;
}

// Calculate the cost of the current state
double cost_function(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, double *u_sequence) {
    State next_state = *state;
    double total_cost = 0.0;
    double prev_u = hovorka_params->u_basal;

    for (int i = 0; i < mpc_params->prediction_horizon; i++) {
        double u = u_sequence[i];
        double insulin_rate = (u - prev_u) / 1.0; // 1.0 is the time step

        if (is_constraint_violated(mpc_params, u, insulin_rate)) {
            return DBL_MAX; // Return the maximum representable finite positive double value if a constraint is violated
        }

        hovorka_step(&next_state, hovorka_params, u, 1.0);
        double glucose_error = next_state.glucose - mpc_params->glucose_target; //likely shold replace glucose_setpoint with glucose_target
        double insulin_error = u - hovorka_params->u_basal;
        total_cost += mpc_params->cost_weight * glucose_error * glucose_error + insulin_error * insulin_error;

        prev_u = u;
    }

    return total_cost;
}

// Calculate the modified cost of the current state
double modified_cost_function(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, double *u_sequence) {
    double cost = cost_function(state, hovorka_params, mpc_params, u_sequence);

    State next_state = *state;
    for (int i = 0; i < mpc_params->prediction_horizon; i++) {
        hovorka_step(&next_state, hovorka_params, u_sequence[i], 1.0);
        if (next_state.glucose < mpc_params->glucose_min) {
            cost += 1e6 * pow(mpc_params->glucose_min - next_state.glucose, 2);
        }
    }

    return cost;
}

// Check if the insulin constraint is violated
int is_constraint_violated(MPCParams *mpc_params, double u, double insulin_rate) {
    return u > mpc_params->insulin_max || u < mpc_params->insulin_min ||
           insulin_rate > mpc_params->insulin_rate_max || insulin_rate < mpc_params->insulin_rate_min;
}

// Optimization data structure
typedef struct {
    State *state;
    HovorkaParams *hovorka_params;
    MPCParams *mpc_params;
    int prediction_horizon;
} OptimizationData;

// Wrapper function for the incremental cost function
double cost_function_incremental_wrapper(unsigned n, const double *du_sequence, double *grad, void *data) {
    OptimizationData *opt_data = (OptimizationData *)data;
    State state = *(opt_data->state);
    HovorkaParams *hovorka_params = opt_data->hovorka_params;
    MPCParams *mpc_params = opt_data->mpc_params;

    double cost = 0.0;
    double u = hovorka_params->u_basal;

    for (int i = 0; i < opt_data->prediction_horizon; i++) {
        u += du_sequence[i];
        cost += modified_cost_function(&state, hovorka_params, mpc_params, u);
        hovorka_step(&state, hovorka_params, u, 1.0);
    }

    return cost;
}

// Wrapper function for the incremental constraints
void constraints_incremental_wrapper(unsigned m, double *result, unsigned n, const double *du_sequence, double *grad, void *data) {
    MPCParams *mpc_params = (MPCParams *)data;
    double u = mpc_params->u_basal;

    for (unsigned i = 0; i < m; i++) {
        u += du_sequence[i];
        result[i] = mpc_params->insulin_max - u;
        result[m + i] = u - mpc_params->insulin_min;
    }
}

// Perform Gradient Descent Optimization
void gradient_descent(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, double *u_sequence_opt, double initial_learning_rate, int prediction_horizon, int max_iterations, double early_stopping_tolerance, int patience) {
    double *grad_sequence = malloc(prediction_horizon * sizeof(double));
    double *u_sequence_try = malloc(prediction_horizon * sizeof(double));
    double *costs_sequence = malloc(prediction_horizon * sizeof(double));
    double learning_rate = initial_learning_rate;

    // Initialize the insulin sequence
    for (int i = 0; i < prediction_horizon; i++) {
        u_sequence_opt[i] = hovorka_params->u_basal;
    }

    double *m = calloc(prediction_horizon, sizeof(double));
    double *v = calloc(prediction_horizon, sizeof(double));

    int num_no_improvement = 0;
    double best_cost = DBL_MAX;

    // Perform gradient descent
    for (int iter = 0; iter < max_iterations; iter++) {
        State state_copy = *state;
        double cost = cost_function(&state_copy, hovorka_params, mpc_params, u_sequence_opt);

        // Compute the gradients for each time step
        for (int i = 0; i < prediction_horizon; i++) {
            memcpy(u_sequence_try, u_sequence_opt, prediction_horizon * sizeof(double));
            u_sequence_try[i] += GRADIENT_STEP_SIZE;

            // Compute the cost function for the perturbed sequence
            State state_copy_simulation = state_copy;
            for (int k = 0; k < prediction_horizon; k++) {
                hovorka_step(&state_copy_simulation, hovorka_params, u_sequence_try[k], 1.0);
            }
            double cost_try = cost_function(&state_copy_simulation, hovorka_params, mpc_params, u_sequence_try);
            costs_sequence[i] = cost_try;
            // Compute the gradient for the time step
            grad_sequence[i] = (cost_try - cost) / GRADIENT_STEP_SIZE;
            u_sequence_try[i] = u_sequence_opt[i];
        }

        // Update the insulin sequence using the gradients and Adam optimizer
        double norm_grad = 0.0;
        for (int i = 0; i < prediction_horizon; i++) {
            m[i] = BETA1 * m[i] + (1 - BETA1) * grad_sequence[i];
            v[i] = BETA2 * v[i] + (1 - BETA2) * pow(grad_sequence[i], 2);

            double m_hat = m[i] / (1 - pow(BETA1, iter + 1));
            double v_hat = v[i] / (1 - pow(BETA2, iter + 1));

            u_sequence_opt[i] -= learning_rate * m_hat / (sqrt(v_hat) + EPSILON);
            norm_grad += pow(grad_sequence[i], 2);
        }
        norm_grad = sqrt(norm_grad);

        // Check if the cost function has improved
        if (cost < best_cost - early_stopping_tolerance) {
            best_cost = cost;
            num_no_improvement = 0;
        } else {
            num_no_improvement++;
        }
        // Implement early stopping
        if (num_no_improvement >= patience) {
            break;
        }
    }

    free(grad_sequence);
    free(u_sequence_try);
    free(costs_sequence);
    free(m);
    free(v);
}

// Perform Sequential Quadratic Programming optimization using SLSQP
void sqp_optimize(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, int prediction_horizon, double *du_opt) {
    assert(prediction_horizon > 0);

    nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, prediction_horizon);
    OptimizationData opt_data = {state, hovorka_params, mpc_params, prediction_horizon};
    nlopt_set_min_objective(opt, cost_function_incremental_wrapper, &opt_data);

    double *lb = malloc(prediction_horizon * sizeof(double));
    double *ub = malloc(prediction_horizon * sizeof(double));
    if (!lb || !ub) {
        fprintf(stderr, "Error allocating memory for bounds.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < prediction_horizon; i++) {
        lb[i] = mpc_params->insulin_min - hovorka_params->u_basal;
        ub[i] = mpc_params->insulin_max - hovorka_params->u_basal;
    }

    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    nlopt_add_inequality_mconstraint(opt, 2 * prediction_horizon, constraints_incremental_wrapper, mpc_params, NULL);

    nlopt_set_xtol_rel(opt, 1e-4);
    nlopt_set_maxeval(opt, 1000);

    double minf;
    nlopt_result result = nlopt_optimize(opt, du_opt, &minf);

    if (result < 0) {
        fprintf(stderr, "Failed to find a solution: error code %d\n", result);
        exit(EXIT_FAILURE);
    } else {
        printf("Found solution with cost: %0.10g\n", minf);
    }

    nlopt_destroy(opt);
    free(lb);
    free(ub);
}

// Compute the gradient of the cost function
void gradient(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, double *u_sequence_opt, double *grad, int prediction_horizon) {
    State state_copy;
    double epsilon = 1e-5;
    for (int i = 0; i < prediction_horizon; i++) {
        state_copy = *state;
        double cost_before = modified_cost_function(&state_copy, hovorka_params, mpc_params, u_sequence_opt[i]);
        u_sequence_opt[i] += epsilon;
        state_copy = *state;
        double cost_after = modified_cost_function(&state_copy, hovorka_params, mpc_params, u_sequence_opt[i]);
        u_sequence_opt[i] -= epsilon;
        grad[i] = (cost_after - cost_before) / epsilon;
    }
}
// Perform Adam Optimization
void adam_optimize(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, double *u_sequence_opt, double learning_rate, double beta1, double beta2, double epsilon, int prediction_horizon, int max_iterations) {

    int num_params = prediction_horizon;
    double *m = calloc(num_params, sizeof(double));
    double *v = calloc(num_params, sizeof(double));
    double *grad = calloc(num_params, sizeof(double));

    for (int t = 1; t <= max_iterations; ++t) {
        gradient(state, hovorka_params, mpc_params, u_sequence_opt, grad, prediction_horizon);

        for (int i = 0; i < num_params; ++i) {
            m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];

            double m_hat = m[i] / (1 - pow(beta1, t));
            double v_hat = v[i] / (1 - pow(beta2, t));

            u_sequence_opt[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }

    free(m);
    free(v);
    free(grad);
}

// Find the optimal insulin increment using the incremental approach
typedef double (*OptimizationAlgorithm)(State *, HovorkaParams *, MPCParams *, int); // Function pointer type
OptimizationAlgorithm active_algorithm; // Pointer to the active optimization algorithm

void switch_optimization_algorithm(OptimizationAlgorithm new_algorithm) { // Switch the optimization algorithm
    active_algorithm = new_algorithm; // Set the active algorithm to the new algorithm
}

double find_optimal_insulin_incremental(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, int prediction_horizon) {
    double *du_opt = malloc(prediction_horizon * sizeof(double));
    if (!du_opt) {
        fprintf(stderr, "Error allocating memory for du_opt.\n");
        exit(EXIT_FAILURE);
    }

    sqp_optimize(state, hovorka_params, mpc_params, prediction_horizon, du_opt);

    double optimal_insulin = hovorka_params->u_basal + du_opt[0];
    free(du_opt);

    return optimal_insulin;
}

// Find the optimal insulin increment using the gradient descent approach
double find_optimal_insulin_gradient_descent(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, int prediction_horizon, int max_iterations) {

    double *u_sequence_opt = malloc(prediction_horizon * sizeof(double));
    gradient_descent(state, hovorka_params, mpc_params, u_sequence_opt, 0.01, prediction_horizon, max_iterations);

    double optimal_insulin = u_sequence_opt[0];
    free(u_sequence_opt);

    return optimal_insulin;
}

// Find the optimal insulin increment using the Interior Point Method (IPM)
double find_optimal_insulin_ipm(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, int prediction_horizon) {
    double *du_opt = malloc(prediction_horizon * sizeof(double));
    if (!du_opt) {
        fprintf(stderr, "Error allocating memory for du_opt.\n");
        exit(EXIT_FAILURE);
    }

    nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, prediction_horizon); // Replace NLOPT_LD_MMA with a suitable IPM algorithm
    OptimizationData opt_data = {state, hovorka_params, mpc_params, prediction_horizon};
    nlopt_set_min_objective(opt, cost_function_incremental_wrapper, &opt_data);

    double *lb = malloc(prediction_horizon * sizeof(double));
    double *ub = malloc(prediction_horizon * sizeof(double));
    if (!lb || !ub) {
        fprintf(stderr, "Error allocating memory for bounds.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < prediction_horizon; i++) {
        lb[i] = mpc_params->insulin_min - hovorka_params->u_basal;
        ub[i] = mpc_params->insulin_max - hovorka_params->u_basal;
    }

    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    nlopt_add_inequality_mconstraint(opt, 2 * prediction_horizon, constraints_incremental_wrapper, mpc_params, NULL);

    nlopt_set_xtol_rel(opt, 1e-4);
    nlopt_set_maxeval(opt, 1000);

    double minf;
    nlopt_result result = nlopt_optimize(opt, du_opt, &minf);

    if (result < 0) {
        fprintf(stderr, "Failed to find a solution: error code %d\n", result);
        exit(EXIT_FAILURE);
    } else {
        printf("Found solution with cost: %0.10g\n", minf);
    }

    nlopt_destroy(opt);
    free(lb);
    free(ub);

    double optimal_insulin = hovorka_params->u_basal + du_opt[0];
    free(du_opt);

    return optimal_insulin;
}

// Find the optimal insulin increment using the gradient descent approach with Adam optimizer
double find_optimal_insulin_adam(State *state, HovorkaParams *hovorka_params, MPCParams *mpc_params, int prediction_horizon, int max_iterations) {
    double *u_sequence_opt = malloc(prediction_horizon * sizeof(double));

    // Initialize u_sequence_opt with basal insulin values
    for (int i = 0; i < prediction_horizon; i++) {
        u_sequence_opt[i] = hovorka_params->u_basal;
    }

    adam_optimize(state, hovorka_params, mpc_params, u_sequence_opt, 0.01, 0.9, 0.999, 1e-8, prediction_horizon, max_iterations);

    double optimal_insulin = u_sequence_opt[0];
    free(u_sequence_opt);

    return optimal_insulin;
}

#define TIME_STEP 1.0

OptimizationAlgorithm gradient_descent_algorithm = {
    .find_optimal_insulin_fn = find_optimal_insulin_gradient_descent
};

OptimizationAlgorithm sqp_algorithm = {
    .find_optimal_insulin_fn = find_optimal_insulin_incremental
};

OptimizationAlgorithm ipm_algorithm = {
    .find_optimal_insulin_fn = find_optimal_insulin_ipm
};

OptimizationAlgorithm adam_algorithm = {
    .find_optimal_insulin_fn = find_optimal_insulin_adam
};


int main() {
    // Initialize Hovorka parameters
    double BW = 70.0; // Replace 70.0 with the actual body weight value
    State state = {0.0, 0.0};
    HovorkaParams hovorka_params = hovorka_parameters(BW);

    // Initialize MPC parameters
    MPCParams mpc_params = {
        .glucose_target = 110.0, // Set your desired glucose target
        .glucose_min = 70.0,     // Set the minimum glucose level
        .insulin_min = 0.0,      // Set the minimum insulin level
        .insulin_max = 25.0      // Set the maximum insulin level
    };

    // Initialize DRV2700 sensor and actuator
    DRV2700_init();

    OptimizationAlgorithm *active_algorithm = &sqp_algorithm; // Set the initial active algorithm
    //active_algorithm = find_optimal_insulin_incremental;
    while (1) {
        // Read sensor data
        state.glucose = DRV2700_read_glucose();

        // Read external user command
        uint8_t command = DRV2700_read_command();

            // Switch to a different algorithm, e.g., upon receiving a command
        if (command == 0x01) {
            active_algorithm = &gradient_descent_algorithm;
        } else if (command == 0x02) {
            active_algorithm = &sqp_algorithm;
        } else if (command == 0x03) { // Add this block to handle command == 0x04
            active_algorithm = &ipm_algorithm;
        } else if (command == 0x04) { // Add this block to handle command == 0x08
            active_algorithm = &adam_algorithm;
        }
  
        state.insulin = DRV2700_read_insulin();

        // Compute control action using active algorithm
        double insulin_control = active_algorithm->find_optimal_insulin_fn(&state, &hovorka_params, &mpc_params, 10); // third method tried
 

        // Apply control action
        if (!is_constraint_violated(&mpc_params, insulin_control)) {
            DRV2700_set_insulin(insulin_control);
        }

        // Wait for the next control step
        delay(TIME_STEP);
    }

    return 0;
}
