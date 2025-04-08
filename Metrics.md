Here are the *formulas* for the evaluation metrics listed in your document:

---

### *1. Noise Reduction / Image Quality Metrics*

- *MSE (Mean Squared Error)*  
  ```math
  \text{MSE} = \frac{1}{MN} \sum_{i=1}^{M} \sum_{j=1}^{N} [I(i,j) - K(i,j)]^2
  ```  
  where \( I \) is the original image, \( K \) is the processed image.

- *RMSE (Root Mean Squared Error)*  
  ```math
  \text{RMSE} = \sqrt{\text{MSE}}
  ```

- *PSNR (Peak Signal-to-Noise Ratio)*  
  ```math
  \text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX^2}{\text{MSE}}\right)
  ```  
  where \( MAX \) is the maximum possible pixel value (e.g., 255).

- *SSIM (Structural Similarity Index)*  
  ```math
  \text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
  ```

---

### *2. Segmentation Metrics*

- *Jaccard Index (IoU)*  
  ```math
  \text{IoU} = \frac{|A \cap B|}{|A \cup B|}
  ```

- *Dice Coefficient (F1 Score)*  
  ```math
  \text{Dice} = \frac{2|A \cap B|}{|A| + |B|}
  ```

- *Rand Index (RI)*  
  ```math
  \text{RI} = \frac{TP + TN}{TP + FP + FN + TN}
  ```

- *Precision & Recall*  
  ```math
  \text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
  ```

- *Hausdorff Distance*  
  ```math
  H(A, B) = \max\left\{\sup_{a\in A} \inf_{b\in B} d(a,b), \sup_{b\in B} \inf_{a\in A} d(a,b) \right\}
  ```

- *Pixel Accuracy*  
  ```math
  \text{Accuracy} = \frac{\text{Number of correctly classified pixels}}{\text{Total number of pixels}}
  ```

---

### *3. Edge Detection Metrics*

- *Pratt's Figure of Merit (FOM)*  
  ```math
  \text{FOM} = \frac{1}{\max(N_I, N_D)} \sum_{i=1}^{N_D} \frac{1}{1 + \alpha d_i^2}
  ```

- *Edge Preservation Index (EPI)*  
  ```math
  \text{EPI} = \frac{\sum (|\nabla I_o| \cdot |\nabla I_f|)}{\sum (|\nabla I_o|^2)}
  ```  
  where \( \nabla I \) is the image gradient, \( I_o \): original, \( I_f \): filtered.

---

### *4. Enhancement Metrics*

- *Contrast Improvement Index (CII)*  
  ```math
  \text{CII} = \frac{\text{std}(I_\text{enhanced})}{\text{std}(I_\text{original})}
  ```

- *Entropy*  
  ```math
  H = -\sum p(i) \log_2 p(i)
  ```  
  where \( p(i) \) is the probability of intensity level \( i \).

- *Tenengrad (Focus Measure)*  
  ```math
  T = \sum_{x,y} (G_x^2 + G_y^2)
  ```  
  (computed using Sobel gradients)

- *Histogram Spread (Standard Deviation)*  
  ```math
  \sigma = \sqrt{\frac{1}{N} \sum (x_i - \mu)^2}
  ```

- *AMBE (Absolute Mean Brightness Error)*  
  ```math
  \text{AMBE} = |\mu_{\text{original}} - \mu_{\text{enhanced}}|
  ```

---

### *5. Color Image Evaluation Metrics*

- *CIEDE2000 / CIELAB ΔE*  
  (Complex formula; generally calculated using color science libraries)

- *Colorfulness Metric*  
  ```math
  \sigma_{rg} = \sqrt{\text{var}(R-G)}, \quad \sigma_{yb} = \sqrt{\text{var}(0.5(R+G) - B)}
  ```
  ```math
  \text{Colorfulness} = \sqrt{\sigma_{rg}^2 + \sigma_{yb}^2} + 0.3 \cdot \sqrt{\mu_{rg}^2 + \mu_{yb}^2}
  ```

- *Color Accuracy (MAE)*  
  ```math
  \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |C_{i}^{\text{original}} - C_{i}^{\text{enhanced}}|
  ```

---

### *6. Compression Quality Metrics*

- *Compression Ratio*  
  ```math
  \text{CR} = \frac{\text{Original Size}}{\text{Compressed Size}}
  ```

- *Bitrate (Bits Per Pixel)*  
  ```math
  \text{bpp} = \frac{\text{Total bits}}{\text{Number of pixels}}
  ```

- *MOS (Mean Opinion Score)*  
  Subjective score typically from 1 (bad) to 5 (excellent), based on human assessment.

---

### *7. Registration & Alignment Metrics*

- *Mutual Information (MI)*  
  ```math
  MI(X, Y) = \sum_{x,y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)
  ```

- *Normalized Cross-Correlation (NCC)*  
  ```math
  \text{NCC} = \frac{\sum (I_1 - \bar{I}_1)(I_2 - \bar{I}_2)}{\sqrt{\sum (I_1 - \bar{I}_1)^2 \sum (I_2 - \bar{I}_2)^2}}
  ```

- *Target Registration Error (TRE)*  

  ```math
  \text{TRE} = \|p_{\text{true}} - p_{\text{registered}}\|
  ```

---

### *8. Computational Metrics*

- *Execution Time*: Measured using time.time() or profiling tools.  
- *Memory Usage*: Measured via libraries like psutil.  
- *Number of Iterations*: Count of steps for convergence in iterative algorithms.

---

Let me know if you'd like these compiled into a PDF/LaTeX table or explained with sample images/code!