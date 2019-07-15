---
title: "Another Algorithm for Optimization : Newton's Method"
layout: post
date: 2019-07-11 11:00

#image: /assets/images/nmlr/
headerImage: false
tag: [deeplearning, machinelearning, logisticregression]
category: blog
author: kaushkay
description: This algorithm can be used in the place of "Simple Logistic Regression".
---

<div style="text-align:center"><img src="/assets/images/blogs/nmlr/NewtonIteration_Ani.gif"></div>

## What is the Classification Problem ?
In Machine Learning and statistics, classification is the problem of detecting / identifying to which of a set of categories (or sub-populations) a new observation belongs, on the basis of the training set of the data containing obseravations (or instances) whose category membership (or labels) are known. Examples of Classification Problem are assigning a given email to the "spam" or "non-spam" class,and assigning a diagnosis to a given patient based on observed characteristics of the patient (sex, blood pressure, presence or absence of certain symptoms, etc.). Classification is an example of pattern recognition.

In Machine Learning, classification is cosidered an instance of supervised learning, i.e., learning where a training set of correctly identified observations is available.

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense for h(x) to take values larger than 1 or smaller than 0 when we know that y ∈ {0, 1}.

To fix this, let’s change the form for our hypotheses h(x). We will choose

<div style="text-align:center">h(x) = g((θ^T) * x) =1/(1 + (e^(-T))*x)</div>

where 
<div style="text-align:center">g(z) = 1 / (1 + e^(−z))</div>

is called the logistic function or the sigmoid function.


The Newton's Method can be used in place of Simple Logistic Regression Algorithm. It can be used to maximize the log likelihood function, l(θ).






## References :
1. <http://mathworld.wolfram.com/NewtonsMethod.html>
2. <https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization>
3. <https://www.softcover.io/read/bf34ea25/math_for_finance/multivariable_methods>
4. <https://en.wikipedia.org/wiki/Newton%27s_method>

---
