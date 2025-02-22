Harris响应公式如下：
\[ R = \text{det}(M) - k \cdot (\text{trace}(M))^2 \]

其中，矩阵 \( M \) 是图像梯度的二阶矩阵，定义为：
\[ M = \begin{bmatrix}
I_{xx} & I_{xy} \\
I_{xy} & I_{yy}
\end{bmatrix} \]

- \(\text{det}(M) = I_{xx} \cdot I_{yy} - I_{xy} \cdot I_{xy}\) 是矩阵的行列式。
- \(\text{trace}(M) = I_{xx} + I_{yy}\) 是矩阵的迹。
- \( k \) 是一个经验参数，通常取值在0.04到0.06之间。