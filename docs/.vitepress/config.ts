import { defineConfig } from "vitepress";

export default defineConfig({
  title: "protoBanana",
  description: "OSS chat-native image generation + editing — open-source counterpart to Nano-Banana 2 / GPT-Image-2",
  // Project pages live at protoLabsAI.github.io/protoBanana/
  base: "/protoBanana/",
  cleanUrls: true,
  lastUpdated: true,
  // The migrated markdown still has cross-doc links from the pre-VitePress
  // era (e.g. [INSTALLATION.md](INSTALLATION.md), absolute repo paths,
  // etc.). Surface them as warnings instead of build failures; a follow-up
  // PR audits + rewrites links to VitePress-style /paths.
  ignoreDeadLinks: true,

  head: [
    ["meta", { name: "theme-color", content: "#facc15" }],
    ["meta", { property: "og:type", content: "website" }],
    ["meta", { property: "og:title", content: "protoBanana — OSS chat-native image gen + edit" }],
    ["meta", { property: "og:description", content: "Open-source counterpart to Google's Nano-Banana 2 / OpenAI's GPT-Image-2, served as an OpenAI-compatible LiteLLM provider on top of ComfyUI." }],
  ],

  // -----------------------------------------------------------------
  // Sidebar is organised by Diátaxis (https://diataxis.fr): four
  // quadrants, four user intents.
  //
  //   Tutorials   → learning-oriented (start here, hands-on path)
  //   How-to      → task-oriented ("how do I X?")
  //   Reference   → information-oriented (lookup tables, schemas)
  //   Explanation → understanding-oriented (the why)
  //
  // Existing pages are mapped into these quadrants without renaming
  // the files (preserves URLs + the auto-copied deep-dives/ tree).
  // The /diataxis page summarises the map; gaps (e.g. tutorials
  // beyond the quickstart) are flagged there.
  // -----------------------------------------------------------------

  themeConfig: {
    logo: "🍌",
    siteTitle: "protoBanana",

    nav: [
      { text: "Tutorials", link: "/guide/quickstart" },
      { text: "How-to", link: "/installation" },
      { text: "Reference", link: "/api" },
      { text: "Explanation", link: "/architecture" },
      { text: "0.1.0a", items: [
        { text: "Changelog", link: "/deep-dives/changelog" },
        { text: "Diátaxis map", link: "/diataxis" },
        { text: "GitHub", link: "https://github.com/protoLabsAI/protoBanana" },
      ]},
    ],

    sidebar: {
      // Default site sidebar: full diátaxis layout, all four quadrants
      // visible at once. Used on /, /agent, /api, etc.
      "/": [
        {
          text: "📘 Tutorials — start here",
          items: [
            { text: "Quickstart (5 min)", link: "/guide/quickstart" },
          ],
        },
        {
          text: "🛠 How-to guides — get a thing done",
          items: [
            { text: "Install protoBanana into a gateway", link: "/installation" },
            { text: "Operate day-2", link: "/operating" },
            { text: "Add a new ComfyUI workflow", link: "/workflows-cookbook" },
            { text: "Validate workflows before shipping", link: "/validating-workflows" },
            { text: "Enable the chat agent", link: "/agent" },
            { text: "Enable Langfuse tracing", link: "/observability" },
            { text: "Run the Gradio test/eval UI", link: "/gradio-app" },
          ],
        },
        {
          text: "📑 Reference — lookup",
          items: [
            { text: "API (endpoints, request shapes)", link: "/api" },
            { text: "Architecture (component map)", link: "/architecture" },
            { text: "Keyword intent router (fallback)", link: "/intent-router" },
            { text: "Benchmarks", link: "/benchmarks" },
          ],
        },
        {
          text: "💡 Explanation — the why",
          items: [
            { text: "Diátaxis map of these docs", link: "/diataxis" },
            { text: "Proposal — strategic system design", link: "/deep-dives/proposal" },
            { text: "Phases — what shipped + why", link: "/deep-dives/phases" },
            { text: "Journey — how we got here", link: "/deep-dives/journey" },
            { text: "Decisions (ADRs)", link: "/deep-dives/decisions" },
            { text: "Changelog", link: "/deep-dives/changelog" },
          ],
        },
      ],
      // Tutorial sidebar: focused, less noise, while the user is
      // walking through the quickstart.
      "/guide/": [
        {
          text: "📘 Tutorials",
          items: [
            { text: "Quickstart", link: "/guide/quickstart" },
          ],
        },
        {
          text: "Next steps (how-to)",
          items: [
            { text: "Full install", link: "/installation" },
            { text: "Enable the chat agent", link: "/agent" },
            { text: "Run the Gradio app", link: "/gradio-app" },
          ],
        },
      ],
      // Deep-dives view: explanation-heavy, for when the user is
      // trying to understand decisions vs. follow steps.
      "/deep-dives/": [
        {
          text: "💡 Explanation",
          items: [
            { text: "Proposal", link: "/deep-dives/proposal" },
            { text: "Phases roadmap", link: "/deep-dives/phases" },
            { text: "Journey", link: "/deep-dives/journey" },
            { text: "Decisions (ADRs)", link: "/deep-dives/decisions" },
            { text: "Diátaxis map", link: "/diataxis" },
          ],
        },
        {
          text: "🛠 How-to (recipes)",
          items: [
            { text: "User-facing recipes", link: "/deep-dives/howto" },
          ],
        },
        {
          text: "📑 Reference",
          items: [
            { text: "Changelog", link: "/deep-dives/changelog" },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/protoLabsAI/protoBanana" },
    ],

    footer: {
      message: "Apache-2.0 licensed. Docs follow the <a href='https://diataxis.fr'>Diátaxis</a> framework.",
      copyright: "© 2026 protoLabsAI · Built by humans + Claude. Fork on <a href='https://github.com/protoLabsAI/protoBanana'>GitHub</a>.",
    },

    editLink: {
      pattern: "https://github.com/protoLabsAI/protoBanana/edit/main/docs/:path",
      text: "Edit this page on GitHub",
    },

    search: {
      provider: "local",
    },
  },
});
