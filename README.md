# NCA-LLM

# Project Timeline
* 2023.12.06 Basic Model setup
* 2023.12.07 
  - decided to start from simple data: "aa bb cc"
  - fix the bug in live mask
  - test.py couldn't run properly due to the structure of the model
* 2023.12.12
  - the new NCA-LLM model could re-generate Shakespeare 
    - implement embedding table on the model
    - use con1d to learn neighbors' logit
  - simplified transformer model tested
 
* 2024.01.03
  - added a group convolution layer acts as filter
  - created word level and character level embedding at two branches respectively
  - [Overleaf Link](https://www.overleaf.com/project/65824d2027c1cfc95ee60a6d)
