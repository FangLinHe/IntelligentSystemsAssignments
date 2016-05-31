## Description:
This assignment is about **neural networks** implementations with two questions.  I used Matlab to implement solutions.  You can see the question descriptions and my answers in [Assignment1_FanglinHe.pdf](Assignment1_FanglinHe.pdf).

----------------------------------------------------------------------------

### Q1 - how to run the code
[Q1.m](Q1.m) is the main script to train the single neural network. As stated in the assignment 1, Question 1, [Q1.m](Q1.m) is composed of four parts which are blocked with %%:
- Q1 (a)
- Q1 (b)(c)
- Q1 (d)
- Q1 (e)

They are corresponding to the assignment 1(a)~(e).

After execution, totally eight figures are produced:
- Figure 1, for Q1(a), plots the initial separation line.
- Figure 2, for Q1(b)(c), plots the new separation line after applying delta rule.
- Figure 3, for Q1(d), plots the separation lines during the iterations of training, until all the data are correctly classified.
- Figure 4, for Q1(d), plots the error values during iterations.
- Figure 5, for Q1(d), plots the final separation line after training.
- Figure 6, for Q1(e), plots a better training strategy and its separation lines during the iterations.
- Figure 7, for Q1(e), plots the error values during iterations.
- Figure 8, for Q1(e), plots the final separation line after training.

To train the neural network, I implemented a function [single_neural.m](single_neural.m) with some parameters, and called this function with different parameters according to the requirements.

----------------------------------------------------------------------------

### Q2 - how to run the code
[Q2.m](Q2.m) is the main script to train the multi-layer neural network. For convenience, [Q2.m](Q2.m) will call the scripts [Q2_a.m](Q2_a.m), [Q2_b.m](Q2_b.m), ..., [Q2_e.m](Q2_e.m).  When you want to execute only one part, you can comment the other parts with %. Similar to [Q1.m](Q1.m), I implemented a function [multiple_neural.m](multiple_neural.m) with several parameters and called this function with different parameters according to the requirements.
