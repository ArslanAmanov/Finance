import math
import random


def exact_binomial_test_two_sided(heads: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial p-value."""
    if not (0 <= heads <= n):
        raise ValueError("heads must be between 0 and n.")
    probs = [math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
            for k in range(n + 1)]
    p_obs = probs[heads]
    p_value = sum(p for p in probs if p <= p_obs + 1e-15)
    return min(1.0, p_value)


def wilson_ci(phat: float, n: int, z: float = 1.96):
    """Approximate 95% Wilson confidence interval for proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z / denom) * \
        math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def parse_outcomes(text: str):
    """Parse inputs like: H T H T or H,T,H,T or 1 0 1 0."""
    raw = text.replace(",", " ").split()
    if not raw:
        raise ValueError("No outcomes entered.")
    parsed = []
    for x in raw:
        if x.lower() == "h" or x == "1":
            parsed.append("H")
        elif x.lower() == "t" or x == "0":
            parsed.append("T")
        else:
            raise ValueError(f"Invalid token: {x}. Use H/T or 1/0.")
    return parsed


def run_test(outcomes, alpha=0.05):
    n = len(outcomes)
    heads = sum(1 for x in outcomes if x == "H")
    tails = n - heads
    phat = heads / n
    p_value = exact_binomial_test_two_sided(heads, n, p0=0.5)
    ci_low, ci_high = wilson_ci(phat, n)

    print("\n--- Coin Fairness Report ---")
    print(f"n (flips): {n}")
    print(f"Heads: {heads}")
    print(f"Tails: {tails}")
    print(f"p-hat (heads proportion): {phat:.4f}")
    print(f"Exact two-sided p-value: {p_value:.6f}")
    print(f"95% CI for p (Wilson): [{ci_low:.4f}, {ci_high:.4f}]")
    if p_value < alpha:
        print(f"Decision at alpha={alpha}: Reject H0 (coin likely unfair).")
    else:
        print(
            f"Decision at alpha={alpha}: Fail to reject H0 (consistent with fair coin).")


def main():
    print("Coin Fairness Test")
    print("1) Simulate flips")
    print("2) Enter outcomes manually")
    mode = input("Choose 1 or 2: ").strip()

    if mode == "1":
        n = int(input("How many flips to simulate? ").strip())
        if n <= 0:
            raise ValueError("n must be positive.")
        seed_text = input(
            "Optional random seed (press Enter to skip): ").strip()
        if seed_text:
            random.seed(int(seed_text))
        outcomes = [random.choice(["H", "T"]) for _ in range(n)]
        print("Simulated outcomes generated.")
    elif mode == "2":
        text = input("Enter outcomes (example: H T H T or 1 0 1 0): ").strip()
        outcomes = parse_outcomes(text)
    else:
        raise ValueError("Invalid mode. Choose 1 or 2.")

    run_test(outcomes, alpha=0.05)


if __name__ == "__main__":
    main()
