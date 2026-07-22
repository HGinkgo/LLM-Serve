from random import Random


def poisson_arrival_times(
    num_requests: int,
    request_rate: float,
    seed: int,
) -> list[float]:
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if num_requests == 0:
        return []

    rng = Random(seed)
    arrivals = [0.0]
    for _ in range(1, num_requests):
        arrivals.append(arrivals[-1] + rng.expovariate(request_rate))
    return arrivals
