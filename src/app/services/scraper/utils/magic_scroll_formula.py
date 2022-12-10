def magic_scroll_formula(number_of_reviews: int) -> int:
    result = int(number_of_reviews / 4) + 1
    if result < 20:
        return 20
    if result < 400:
        return result
    return 400