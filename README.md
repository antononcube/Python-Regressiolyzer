# Regressiolyzer

Python package with a class that allows pipeline-like specification and execution of regression workflows.

Extensive guide is given in the Jupyter notebook 
["Regressionizer-demo.ipynb"](https://github.com/antononcube/Python-Regressionizer/blob/main/docs/Regressionizer-demo.ipynb).

-----

## Details

- `Regressionizer` allows for rapid and specification regression workflows.
  - To quickly specify: 
    - data rescaling and summary, 
    - regression computations,
    - outliers finding
    - conditional Cumulative Distribution Functions (CDFs) reconstruction
    - visualization of data, fits, residual errors, outliers, CDFs 

- `Regressionizer` works with data frames, numpy arrays, and lists of numbers, and lists of numeric pairs.

- The curves computed with quantile regression are called **regression quantiles**.

- `Regressionizer` has three regression methods:
  - `quantile_regression`
  - `quantile_regression_fit`
  - `least_squares_fit`
  
- The regression quantiles computed with the method `quantile_regression` correspond to the specified probabilities
  - The regression quantiles are linear combinations of B-splines generated over the specified knots.

- In other words, the method`quantile_regression` computes fits using a B-spline functions basis.  
  - The basis is specified with the `knots` argument and the option `order`.
  - `order` is 3 by default. 

- The methods `quantile_regession_fit` and `least_squares_fit` require a list of basis functions (to fit with.)

------

## Usage examples 

Import libraries:

```python
from Regressionizer import *
import numpy as np
```

Generate random data:

```python
np.random.seed(0)
x = np.linspace(0, 2, 300)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.4, x.shape)
data = np.column_stack((x, y)
```

Compute quantile regression and make the corresponding plot:

```python
obj = (Regressionizer(data)
       .quantile_regression(knots=8, probs=[0.2, 0.5, 0.8])
       .plot(title="B-splines fit", template="plotly")
       )
```

Show the plot obtained above:

```python
obj.take_value().show()
```

![](https://raw.githubusercontent.com/antononcube/Python-Regressionizer/main/docs/img/random-data-B-spline-rqs.png)

------

## References

### Articles, books

[RK1] Roger Koenker, 
[Quantile Regression](https://books.google.com/books/about/Quantile_Regression.html?id=hdkt7V4NXsgC), 
Cambridge University Press, 2005.

[RK2] Roger Koenker,
["Quantile Regression in R: a vignette"](https://cran.r-project.org/web/packages/quantreg/vignettes/rq.pdf),
(2006),
[CRAN](https://cran.r-project.org/).

[AA1] Anton Antonov,
["A monad for Quantile Regression workflows"](https://github.com/antononcube/MathematicaForPrediction/blob/master/MarkdownDocuments/A-monad-for-Quantile-Regression-workflows.md),
(2018),
[MathematicaForPrediction at GitHub](https://github.com/antononcube/MathematicaForPrediction).

### Packages, paclets

[RKp1] Roger Koenker,
[`quantreg`](https://cran.r-project.org/web/packages/quantreg/index.html),
[CRAN](https://cran.r-project.org/).

[AAp1] Anton Antonov,
[Quantile Regression WL paclet](https://github.com/antononcube/WL-QuantileRegression-paclet),
(2014-2023),
[GitHub/antononcube](https://github.com/antononcube).

[AAp2] Anton Antonov,
[Monadic Quantile Regression WL paclet](https://github.com/antononcube/WL-MonadicQuantileRegression-paclet),
(2018-2024),
[GitHub/antononcube](https://github.com/antononcube).

[AAp3] Anton Antonov,
[`QuantileRegression`](https://resources.wolframcloud.com/FunctionRepository/resources/QuantileRegression),
(2019),
[Wolfram Function Repository](https://resources.wolframcloud.com/FunctionRepository/resources/QuantileRegression).

### Repositories

[AAr1] Anton Antonov,
[DSL::English::QuantileRegressionWorkflows in Raku](https://github.com/antononcube/Raku-DSL-English-QuantileRegressionWorkflows),
(2020),
[GitHub/antononcube](https://github.com/antononcube/Raku-DSL-English-QuantileRegressionWorkflows).

### Videos

[AAv1] Anton Antonov,
["Boston useR! QuantileRegression Workflows 2019-04-18"](https://www.youtube.com/watch?v=a_Dk25xarvE),
(2019),
[Anton Antonov at YouTube](https://www.youtube.com/@AAA4Prediction).

[AAv2] Anton Antonov,
["useR! 2020: How to simplify Machine Learning workflows specifications"](https://www.youtube.com/watch?v=b9Uu7gRF5KY),
(2020),
[R Consortium at YouTube](https://www.youtube.com/channel/UC_R5smHVXRYGhZYDJsnXTwg).