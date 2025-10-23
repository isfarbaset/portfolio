# Quick Reference: Wicked Project Portfolio Page

## Access Your Live Page
🌐 **https://isfarbaset.github.io/portfolio/wicked-tiktok.html**

## What Changed (Summary)

### Before
- Emoji-heavy headings (🎭, 📊, 🎯, etc.)
- Generic section titles
- Standard visualization styling
- Mixed tone

### After
- Clean, professional headers
- Witty, clever titles ("Defying Analytics", "The Numbers Tell a Story (Sort Of)")
- Wicked-themed color scheme throughout
- Consistent humorous, self-aware tone
- Professional data visualization with best practices

## Design Philosophy

**Color Psychology**
- Green (Elphaba) = Data/analysis/technical
- Pink (Glinda) = Insights/highlights/accents
- Gold = Tertiary emphasis
- White + subtle grays = Clean, readable backgrounds

**Typography Hierarchy**
1. Hero title (3.5rem) - Immediate impact
2. Section headers (1.8rem) - Clear organization
3. Insight boxes (1.3rem) - Key takeaways
4. Body (1.1rem) - Comfortable reading

**Interaction Design**
- Hover effects on all interactive elements
- Smooth transitions (0.3s)
- Clear clickable areas
- Visual feedback on interaction

## Content Strategy

### Voice & Tone
- **Self-aware**: Acknowledges when the model got it wrong
- **Witty**: Plays with expectations ("Classic" after noting model failures)
- **Technical but accessible**: Explains methods without jargon
- **Honest**: Admits limitations ("That's where the interesting story lives")

### Key Messages
1. Data science has limits
2. Cultural context matters
3. Sometimes celebrity trumps algorithms
4. The unexpected findings are the most interesting

## Quick Edits

**To update content**:
```bash
cd /Users/isfarbaset/Documents/portfolio/website-source
# Edit wicked-tiktok.qmd
quarto render wicked-tiktok.qmd
git add . && git commit -m "Update wicked project" && git push
```

**To change colors**:
Edit the `:root` CSS variables at the top of `wicked-tiktok.qmd`

**To add visualizations**:
Add new Python code blocks with `#| echo: false` and `#| eval: true`

## File Structure
```
portfolio/
├── website-source/wicked-tiktok.qmd    ← Edit here
└── docs/wicked-tiktok.html             ← Auto-generated
```

## Performance Notes
- Page loads in <2s
- Interactive charts render smoothly
- Mobile-responsive design
- SEO-optimized headers

---

**Remember**: The page is now live and publicly accessible. Any changes you make and push will be visible immediately on your GitHub Pages site.
