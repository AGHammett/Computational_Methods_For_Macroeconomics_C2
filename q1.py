def h(theta, alpha, t, c_o):

    num = 100 * (1 - theta)
    den = (alpha / (1 - t)) * c_o + 1 - theta

    return num / den
