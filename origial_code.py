
def train(input_data, n_max_iterations, width, height): # TODO: there are no docstrings or type hints, we should add those for better readability and maintainability.
    # Initial neighborhood radius.
    sigma_0 = max(width, height) / 2  

    # Initial learning rate.
    alpha_0 = 0.1 # TODO: This is hardcoded, so we can't adjust this value for experimenting.

    # Randomly initialize weights for each neuron in the grid.
    weights = np.random.random((width, height, 3))   # TODO: what if input_data.shape[2] is not 3? This number is hardcoded, and needs to be dynamic based on the input data's shape.

    # Time constant for decay of learning rate and neighborhood radius.
    lambd = n_max_iterations / np.log(sigma_0)  

    # ADDED BY ME: start a timer to measure the time taken every N iterations.
    start_time = time.perf_counter()    

    # Training loop.
    for t in range(n_max_iterations):

        # ADDED BY ME: print elapsed time every 10 iterations, and then restart the clock.
        iterations = 10
        if t % iterations == 0 and t != 0:
            end_time = time.perf_counter()
            print(f"Iteration {t}/{n_max_iterations} - Time taken: {end_time - start_time:.4f} seconds.")
            start_time = time.perf_counter() 

        # Update σ and α.
        sigma_t = sigma_0 * np.exp(-t/lambd)
        alpha_t = alpha_0 * np.exp(-t/lambd)

        # Find the Best Matching Unit (BMU) and update weights.
        for vt in input_data:  # TODO: We can use mini batches or shuffle and sample, instead of iterating through each input vector, to help with performance and convergence.

            # Find the BMU.
            bmu = np.argmin(np.sum((weights - vt) ** 2, axis=2))

            # define the BMU's coordinates.
            bmu_x, bmu_y = np.unravel_index(bmu, (width, height))

            # For each node in the grid, calculate the neighborhood function and update the weights.
            for x in range(width): # TODO: Nested loop is too expensive, we can vectorize this operation and improve performance.
                for y in range(height): # TODO: This calculation does not use the input vector, so could be removed from the loop.

                    # Calculate the euclidean distance from the BMU.
                    di = np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2)) # TODO: no need to calculate the square root, we can just use the squared distance.

                    # Calculate the neighborhood function (theta).
                    theta_t = np.exp(-(di ** 2) / (2*(sigma_t ** 2))) # TODO: distance should be squared, so we can avoid calculating the square root.

                    # Update the weights of the node.
                    weights[x, y] += alpha_t * theta_t * (vt - weights[x, y])

    return weights