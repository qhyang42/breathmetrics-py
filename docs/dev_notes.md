# development notes for breathmetrics python
9/26/2025 scaffolding built. started building helper function. no test performed yet. 
9/27/2025 add find extrema. need to find a testing data 
# pipeline from orig breathmetrics toolbox: 
## preproc
check input. i can take numpy array, CSV, .mat file
baseline correction. demean, zscore 
decide internally which resp to use. mean centered or not 
## feature extraction. all of these go into kernel.py 
find extrema. *wrote, no test*
find onsets and pauses. this will have to be modified to new functions *wrote, no test*
find inhale and exhale offsets 
find pause 
calculate inhale and exhale volume 
calculate secondary features 
calculate resp phase 
print secondary features (util?)
check if features are estimated (util?)
## ERPs 
calculate ERP 
calculate resampled ERP 
## plotting 



---
# markdown syntex cheatsheet 
# Heading 1
## Heading 2
### Heading 3

*italic* or _italic_
**bold** or __bold__
***bold italic***
~~strikethrough~~

- Item 1
- Item 2
  - Nested item

1. First
2. Second
3. Third

[OpenAI](https://openai.com)

![Alt text](https://example.com/image.png)

Inline code: `print("hello")`

Block code:
```python
def greet():
    print("hello")

### Blockquotes
```markdown
> This is a quote

---



