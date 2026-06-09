<div align="center">

<img src="docs/images/isfar-about-circle.png" width="130" alt="Isfar Baset" />

# Isfar Baset

**I turn research-grade models into things people use.**

Data Analyst II @ Shift Digital &nbsp;·&nbsp; MS Data Science, Georgetown ('25) &nbsp;·&nbsp; Northern Virginia

<br/>

[<kbd> <br/> &nbsp;&nbsp;Visit the site ↗&nbsp;&nbsp; <br/> </kbd>](https://isfarbaset.github.io/portfolio/)

<br/>

[LinkedIn](https://www.linkedin.com/in/isfarbaset/) · [GitHub](https://github.com/isfarbaset) · [Instagram](https://www.instagram.com/is.far/) · [Email](mailto:isfar.baset@gmail.com)

</div>

<br/>

## What this is

My portfolio. One hand-built page, ten projects, and a running theme: take something technical, make it useful, explain it like a human. Originally from Dhaka, now based in the DMV, which explains at least two of the projects below.

<br/>

## The work

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="https://isfarbaset.github.io/portfolio/wicked.html"><img src="docs/images/wicked.jpg" alt="Defying Analytics" /></a>
      <h3><a href="https://isfarbaset.github.io/portfolio/wicked.html">Defying Analytics</a></h3>
      <sup>SPOTIFY · PLOTLY · SCIKIT-LEARN</sup>
      <p>20 years of Wicked streaming data and what makes a song stick. Length doesn't matter. Vulnerability does.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://isfarbaset.github.io/portfolio/Group37_final_Report.html"><img src="docs/images/beat-bytes.jpg" alt="Beats and Bytes" /></a>
      <h3><a href="https://isfarbaset.github.io/portfolio/Group37_final_Report.html">Beats and Bytes</a></h3>
      <sup>PYTHON · ML · SPOTIFY API</sup>
      <p>Predictive models on popular music: which audio features predict popularity, and which just look like they do.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://medium.com/@isfarbaset/fmbench-assistant-an-ai-chatbot-for-navigating-foundation-model-benchmarking-with-fmbench-39615ff08161"><img src="docs/images/fmbench.jpg" alt="FMBench Assistant" /></a>
      <h3><a href="https://medium.com/@isfarbaset/fmbench-assistant-an-ai-chatbot-for-navigating-foundation-model-benchmarking-with-fmbench-39615ff08161">FMBench Assistant</a></h3>
      <sup>BEDROCK · LANGGRAPH · LAMBDA</sup>
      <p>A conversational AI that walks you through foundation-model benchmarking instead of making you read the docs.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://isfarbaset.github.io/portfolio/aqi.html"><img src="docs/images/aqi.jpg" alt="Air Quality Intelligence" /></a>
      <h3><a href="https://isfarbaset.github.io/portfolio/aqi.html">Air Quality Intelligence</a></h3>
      <sup>GPT-4 · MCP · ASYNC PYTHON</sup>
      <p>Ask about air quality in any city, get health-aware advice for jogging, cycling, and travel. Built on a custom MCP server.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://isfarbaset.github.io/netflix-recap/"><img src="docs/images/netflix.png" alt="Netflix Recap" /></a>
      <h3><a href="https://isfarbaset.github.io/netflix-recap/">Netflix Recap</a></h3>
      <sup>PYTHON · PANDAS · DASHBOARDS</sup>
      <p>Your Netflix watch history as a personal year-in-review: viewing patterns, binge habits, genre obsessions.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://isfarbaset.github.io/github-insights/"><img src="docs/images/git-insights.png" alt="GitHub Insights" /></a>
      <h3><a href="https://isfarbaset.github.io/github-insights/">GitHub Insights</a></h3>
      <sup>GITHUB API · CANVAS · JS</sup>
      <p>Drop in a username, get a downloadable stats card: lifetime activity, streaks, top repos, personality badges.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://isfarbaset.github.io/almanac/"><img src="docs/images/almanac.jpg" alt="Almanac" /></a>
      <h3><a href="https://isfarbaset.github.io/almanac/">Almanac</a></h3>
      <sup>REACT · INDEXEDDB · PWA</sup>
      <p>A private, local-only menstrual tracker. No accounts, no servers; the data never leaves your browser.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://isfarbaset.github.io/dmv-blooms/"><img src="docs/images/dmv-blooms.jpg" alt="DMV Blooms" /></a>
      <h3><a href="https://isfarbaset.github.io/dmv-blooms/">DMV Blooms</a></h3>
      <sup>JS · INTERACTIVE MAP</sup>
      <p>A seasonal field guide to flowers across DC, Maryland &amp; Virginia: what's peaking now and where to chase it, 28 spots deep.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://isfarbaset.github.io/fall-2024-project-team-29/"><img src="docs/images/us-insights.jpg" alt="US Insights" /></a>
      <h3><a href="https://isfarbaset.github.io/fall-2024-project-team-29/">US Insights</a></h3>
      <sup>NLP · SENTIMENT · REDDIT</sup>
      <p>Sentiment patterns across U.S. states based on Reddit conversations: what each state talks about and how it feels.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://isfarbaset.github.io/story-project/"><img src="docs/images/temp-talk.jpg" alt="Temp Talk" /></a>
      <h3><a href="https://isfarbaset.github.io/story-project/">Temp Talk</a></h3>
      <sup>CLIMATE · TIME SERIES</sup>
      <p>Climate trends across Southeastern Utah's national parks: heat, soil, evaporation, and precipitation, told as a story.</p>
    </td>
  </tr>
</table>

<br/>

## Under the hood

The homepage is a single `docs/index.html`: React and Tailwind off a CDN, JSX compiled in the browser, no build step. Is that how you're supposed to ship React? No. Does it mean the whole site is one file I can edit in any text editor? Yes, and I like it that way.

The older project pages (Wicked, AQI, Beats and Bytes) are rendered with [Quarto](https://quarto.org/) from `website-source/`. GitHub Pages serves everything from `docs/`.

```
portfolio/
├── docs/             ← the live site (GitHub Pages serves this)
│   ├── index.html    ← the homepage, one hand-built file
│   ├── *.html        ← Quarto-rendered project pages
│   └── images/
└── website-source/   ← Quarto source for the older pages
```

> [!WARNING]
> Running `quarto render` in `website-source/` will overwrite `docs/index.html` with the old Quarto homepage. Don't. (Note to self, mostly.)

## Run it locally

```bash
python3 -m http.server 8000 --directory docs
```

Then open [localhost:8000](http://localhost:8000). That's it. That's the build process.

<br/>

<div align="center">

If you read this far, we should probably talk: [isfar.baset@gmail.com](mailto:isfar.baset@gmail.com)

© 2026 Isfar Baset

</div>
