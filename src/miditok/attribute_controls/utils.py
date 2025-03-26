"""Useful methods for attribute controls."""

from fractions import Fraction
from math import isnan


def valid(x):
    return ((not isnan(x)) and (x != 0))

def prime_factors(n):
    """Compute the prime factors of a number with their powers."""
    i = 2
    factors = {}
    while i * i <= n:
        while (n % i) == 0:
            if i in factors:
                factors[i] += 1
            else:
                factors[i] = 1
            n //= i
        i += 1
    if n > 1:
        factors[n] = 1
    return factors

def indigestibility(n):
    """Calculate the indigestibility xi(N) of a number."""
    factors = prime_factors(n)
    return sum((p**2 * n**2) for p, n in factors.items())

def indispensability_factor(position, bar_length):
    """Calculate indispensability based on the position in the bar."""
    return 1 - (position / bar_length)

def harmonicity_with_indispensability(pitch1, pitch2, position1, position2, bar_length):
    """Calculate harmonicity considering indigestibility and indispensability."""
    # Calculate pitch ratio and indigestibility
    ratio = Fraction(2 ** ((pitch2 - pitch1) / 12)).limit_denominator()
    numerator, denominator = ratio.numerator, ratio.denominator
    indigestibility_sum = indigestibility(numerator) + indigestibility(denominator)

    # Calculate indispensability
    indispensability = indispensability_factor(position1, bar_length) + indispensability_factor(position2, bar_length)

    # Combine into harmonicity
    total_complexity = indigestibility_sum + indispensability
    return 1 / total_complexity if total_complexity > 0 else 0