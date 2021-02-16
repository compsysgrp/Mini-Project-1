## Running the code
For the purposes of testing the code it is reccomended that you use the following line to compile the testmain.c file:

```gcc -O2 -march=native -lm matrix.c -o matrix -lm```

The code has been simplified as much as possible in order to make it more readable and to make testing easier. 
Once you run that file with the command:

```./matrix```

You will be prompted to give arguments regarding matrix dimensions, data type, multiplication method, and whether or not you wish to print. This allows for as much custom testing as needed as opposed to giving you files with set matrices that are guaranteed to work every time. If you have any questions about how to run it, or if you run into any issues, feel free to contact D.J. Bucciero or Annie Tao through webex.

## Data Structure
The "matrices" used are actually one dimensional, not two dimensional. If the desired matrix is 10x10 it would be a 1d array 100 long. The variables for rows and columns are stored so that the data can be parsed effectively and used for multiplication, printing, etc. 

## Multiplication Functions:
### Traditional Multiplication:
Uses three nested "for" loops.  In the deepest loop, the values in the two inputted matrices are multipled and stored into the output matrix. The indexes of the matrices are called by using square brackets "[]".
### SIMD Multiplication:
Similar to the traditional multiplication, the SIMD multiplication uses three nested "for" loops. In the deepest loop, the value of the output matrix and the two input matrices are stored. The two matrices values are multipled and added to the output matrix value. This final value is than stored into the output matrix. SIMD programming is used to read, perform arithematic, and store the values. 
### Cache Optimized "Block" Multiplication:
Block multiplication utiliizes two additional "for" loops and an extra value, "block size." This block size is used in incrementation of certain loops and calculating the bounds of certain loops. The block size is to be a square (N x N) and needs to be smaller than the size of any of the matrices that will be multipled. The arithematic for calculating and storing the values are the same as the traditional multiplication, but by adding the two extra loops and changing the incrementation interval and loop bounds, the run-time of the block multiplication will be shorter.


## Test Results
All tests were were done using a computer that had an i7-8700k processor overclocked to 4.45Ghz.
There was also data collected for sizes 10x10 and 100x100 but the difference in times were negligble, lesser than 1 microsecond.
| Matrix Size   | Traditional int time (s) | Traditional float time (s) | SIMD int time (s) | SIMD float time (s) | Block int time (s) | block float time (s) |
| ------------- |:------------------------:|:--------------------------:|:-----------------:|:-------------------:|:------------------:|:--------------------:|
| 1000x1000     |.750                      |.984                        |2.171              |2.375                |1.093               |1.984                 |
| 2000x2000     |25.968                    |23.031                      |26.218             |28.250               |9.343               |16.671                |
| 3000x3000     |96.015                    |96.515                      |93.281             |100.796              |32.734              |58.187                |
| 4000x4000     |233.687                   |252.78                      |226.796            |233.062              |78.156              |141.531               |
| 5000x5000     |471.859s                  |528.625                     |450.218            |465.218              |153.0               |276.578               |

|Matrix Size|Block Size|
|-----------|:--------:|
|1000x1000  | 100      |
|2000x2000  | 200      |
|3000x3000  | 300      |
|4000x4000  | 400      |
|5000x5000  | 500      |

We expected traditional to be the slowest by far, except at perhaps low numbers, SIMD to be very quick especially at higher numbers, and block to be slower than SIMD but have the benefit that it can work on any processor. Based on our results, block multiplication (cache optimized) was far superior to the others, we are unsure if this is because we programmed the block functions effectively, or if SIMD/traditional was one inefficiently.

![image](https://cdn.discordapp.com/attachments/804497070534033428/811055551546130493/042794c93b88a5e3215759f52b315313.png)

![image](https://cdn.discordapp.com/attachments/804497070534033428/811055566221738014/66a74da027d9ae59253f35d8e9e5ff5e.png)

## Conclusions
Given the issues we had programming the SIMD multiplication aspect of the project we feel that it could've definitely been slightly better. We feel the block and traditional were right on point, and we gave the user plenty of options when it came to playing around with the code. In terms of our understanding of the actual multiplication operations, it was a great chance to get a hands on experience learning how to truly optimize our code even though it can feel clunky or overbearing sometimes. SIMD is definitely a good choice especially for very large numbers, but not a good choice if you're running your code on multiple devices with different processors. Block is a great solution when efficiency is still important but processors vary between devices. Traditional is still a very viable solution for smaller datasets when a quick solution is needed as the differences in smaller matrices are negligble. 
