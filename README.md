Recommendations for Using This Script

### **Model Selection**
- **DeepSeek-R1** (default) - Excellent for complex literary translation requiring deep reasoning about style and nuance
- **Qwen3** - Good alternative with thinking capabilities, faster for simpler texts
- Consider the model size based on your hardware:
  - **8B models** for consumer GPUs (RTX 3060+)
  - **32B+ models** for professional setups with more VRAM

### **Optimal Configuration Settings**

1. **Temperature (0.7 default)**
   - **0.5-0.6** for technical or historical texts requiring accuracy
   - **0.7-0.8** for literary fiction (good balance)
   - **0.8-0.9** for poetry or highly stylistic prose

2. **Page Size**
   - **300-500 words** for dense philosophical texts (Kant, Hegel)
   - **500-800 words** for narrative fiction
   - **200-400 words** for poetry or highly stylized prose

3. **Author Style Matching**
   - **Knausgård** - Best for diary-like texts, memoirs, autobiographical works
   - **Cusk** - Ideal for philosophical dialogues, observational narratives
   - **Ernaux** - Perfect for sociological texts, class-conscious narratives
   - **Sebald** - Excellent for historical fiction, travelogues with melancholic tone

### **Text Preparation Tips**

1. **Pre-process Gutenberg files**:
   ```bash
   # Remove Gutenberg headers/footers
   sed -n '/\*\*\* START OF THIS PROJECT/,/\*\*\* END OF THIS PROJECT/p' input.txt > clean.txt
   ```

2. **For best results**, convert HTML/EPUB to plain text first:
   ```bash
   pandoc input.epub -t plain -o input.txt
   ```

### **Performance Optimization**

1. **GPU Memory Management**:
   - Close other applications using VRAM
   - Use smaller model variants if encountering OOM errors
   - Increase `DELAY_BETWEEN_PAGES` to prevent thermal throttling

2. **Processing Strategy**:
   - Run in test mode first (`--test --pages 10`) to verify style quality
   - Process challenging sections separately with adjusted prompts
   - Use thinking logs to debug translation issues

### **Quality Assurance**

1. **Review thinking logs** - They reveal the model's reasoning about style choices
2. **Compare multiple runs** with different temperatures for critical passages
3. **Post-process** for consistency in:
   - Character names (maintain original or fully translate)
   - Place names and cultural references
   - Formatting of dialogue and internal monologue

### **Recommended Workflow**

1. **Initial Test** (5-10 pages):
   ```bash
   python translator.py source.txt --test --pages 10 --author "Rachel Cusk"
   ```

2. **Adjust parameters** based on test output quality

3. **Full translation** with optimized settings:
   ```bash
   python translator.py source.txt --model deepseek-r1 --author "Karl Ove Knausgård"
   ```

4. **Post-processing**:
   - Review for consistency
   - Merge chapter breaks if needed
   - Final editing pass for flow

### **Advanced Usage**

- **Extend author styles** by adding entries to `AUTHOR_STYLES` dictionary
- **Chain multiple models**: Use thinking model for complex passages, faster model for simple narrative
- **Implement glossary**: Add consistent translation rules for recurring terms
- **Version control**: Save different temperature/style attempts for comparison

### **Common Pitfalls to Avoid**

1. Don't process texts with heavy formatting (footnotes, tables) without pre-processing
2. Avoid very large page sizes (>1000 words) - quality degrades
3. Don't skip the thinking mode - it significantly improves style consistency
4. Remember to check copyright status - ensure texts are truly public domain

The script is designed to be both powerful and flexible. Start with test runs, iterate on settings, and gradually refine based on the specific text characteristics and your quality requirements.
