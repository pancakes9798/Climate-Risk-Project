"""Climate transition risk in the term structure and in corporate credit spreads.

The package is organised in three layers that mirror the methodology:

* :mod:`climate_risk.term_structure` – shifting-endpoint dynamic Nelson–Siegel model of the
  US Treasury zero-coupon curve, with SSP-scenario-conditional endpoints;
* :mod:`climate_risk.credit` – CDS-implied default probabilities and the Battiston–Monasterolo
  climate-policy shock on issuers' asset values (Merton threshold shift);
* :mod:`climate_risk.portfolio` – t-copula default dependence and scenario-conditional
  portfolio loss distributions.
"""

import logging

__version__ = "2.0.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
