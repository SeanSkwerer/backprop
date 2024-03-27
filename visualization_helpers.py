import matplotlib.pyplot as plt
import numpy as np

def plot_errors(errors, use_log=True, baseline_error=0):
    for errs in errors:
        iters = list(range(len(errs)))
        if use_log:
            logerrs = np.log10(np.array(errs)-baseline_error)
            plt.plot(iters, logerrs)
        else:
            plt.plot(iters, errs)
    plt.xlabel("Iterations")
    if use_log:
        plt.ylabel("log error")
    else:
        plt.ylabel("error")
    plt.title('Error per iteration')
    plt.show()