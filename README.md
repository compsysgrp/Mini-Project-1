
## Running the code
For the purposes of testing the code it is reccomended that you use the following line to compile the testmain.c file:

```gcc -O2 -march=native -lm testmain.c -o testmain -lm```

The code has been simplified as much as possible in order to make it more readable and to make testing easier. 
Once you run that file with the command:

```./testmain```

You will be prompted to give arguments regarding matrix dimensions, data type, multiplication method, and whether or not you wish to print. This allows for as much custom testing as needed as opposed to giving you files with set matrices that are guaranteed to work every time. If you have any questions about how to run it, or if you run into any issues, feel free to contact D.J. Bucciero or Annie Tao through webex.

## multiplication functions:
### Traditional multiplication:
blah blah blah
### SIMD multiplication:
blah blah blah
### Cache Optimized "Block" Multiplication:
blah blah blah


## Test Results
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
