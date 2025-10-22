def price_bond(cashflows, r_curve, s_curve):
    """
    Calcola il prezzo teorico di un'obbligazione
    scontando i flussi con curva risk-free e credit spread.

    cashflows: lista di tuple (importo, tempo)
    r_curve: dizionario {tempo: tasso risk-free}
    s_curve: dizionario {tempo: credit spread}
    """
    price = 0
    for amount, t in cashflows:
        r = r_curve.get(t, 0)
        s = s_curve.get(t, 0)
        discount_factor = 1 / ((1 + r + s) ** t)
        price += amount * discount_factor
    return price
