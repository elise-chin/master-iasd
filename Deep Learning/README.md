# Deep Learning

## Course

This course is about using deep learning tools.
The objective of the course is to be able to design deep neural networks and to apply them to various problems. The language used for the course is Torch. It relies on the Lua scipting language augmented with tensor specific instructions. During the course, we will use simple examples to learn how to generate and transform data in Torch as well as how to learn from this data. We will cover deep neural networks, deep convolutional neural networks and some optimizations of the architecture such as residual nets.

## Project

The description of the Deep Learning project can be found at this [link](https://www.lamsade.dauphine.fr/~cazenave/DeepLearningProject.html).

The goal is to train a network for playing the game of Go. In order to be fair about training ressources the number of parameters for the networks you submit must be lower than 100 000. The maximum number of students per team is two. The data used for training comes from the Katago Go program self played games. There are 1 000 000 different games in total in the training set. The input data is composed of 21 19x19 planes (color to play, ladders, current state on two planes, two previous states on four planes). The output targets are the policy (a vector of size 361 with 1.0 for the move played, 0.0 for the other moves), and the value (1.0 if White won, 0.0 if Black won).

This project has been done in collaboration with Noé Lallouet.