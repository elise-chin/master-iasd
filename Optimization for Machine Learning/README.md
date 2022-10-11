# Optimization for Machine Learning

## Course

Optimization is at the heart of most recent advances in machine learning. Indeed, it not only plays a major role in linear regression, SVM and kernel methods, but it is also the key to the recent explosion of deep learning for supervised and unsupervised problems in imaging, vision and natural language processing. This course will review the mathematical foundations, the underlying algorithmic methods and showcase modern applications of a broad range of optimization techniques.

The course will be composed of classical lectures and numerical sessions in Python. It will begin with the basic components of smooth optimization (optimality conditions, gradient-type methods), then move to methods that are particularly relevant in a machine learning setting such as the celebrated stochastic gradient descent algorithm and its variants. More advanced algorithms related to non-smooth and constrained optimization, that encompass known characteristics of learning problems such as the presence of regularizing terms, will also be described. During lab sessions, the various algorithms studied during the lectures will be implemented and tested on real and synthetic datasets: these sessions will also address several practical features of optimization codes such as automatic differentiation, and built-in optimization routines within popular machine learning libraries such as PyTorch.

## Project

The assessment consists of a written exam and a project.

The project is concerned with going beyond first-order techniques in machine learning. Algorithmic frameworks such as gradient descent and stochastic gradient are inherently first-order methods, in that they rely solely on first-order derivatives. Second-order methods, on the other hand, make use of higher-order information, either explicitly or implicitly. Although those techniques are widely used in scientific computing, their use in machine learning has yet to be generalized. The goal of this project is to illustrate the performance of these techniques on learning problems involving both synthetic and real data.
The project is decomposed as follows. In Section 1, we introduce classical optimization schemes, of Newton and quasi-Newton type. We then describe practical variants of these techniques that are more amenable to a machine learning setting: this is the purpose of Section 2. Finally, Section 3
proposes an application of those methods to a binary classification problem based on a real dataset.