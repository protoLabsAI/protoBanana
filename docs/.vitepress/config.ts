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

  themeConfig: {
    logo: "🍌",
    siteTitle: "protoBanana",

    nav: [
      { text: "Guide", link: "/guide/quickstart" },
      { text: "Reference", link: "/api" },
      { text: "Architecture", link: "/architecture" },
      { text: "Deep dives", link: "/deep-dives/proposal" },
      { text: "0.1.0a", items: [
        { text: "Changelog", link: "/deep-dives/changelog" },
        { text: "GitHub", link: "https://github.com/protoLabsAI/protoBanana" },
      ]},
    ],

    sidebar: {
      "/guide/": [
        {
          text: "Getting started",
          items: [
            { text: "Quickstart", link: "/guide/quickstart" },
            { text: "Installation", link: "/installation" },
            { text: "Operating", link: "/operating" },
          ],
        },
        {
          text: "Using protoBanana",
          items: [
            { text: "How-to recipes", link: "/deep-dives/howto" },
            { text: "API reference", link: "/api" },
            { text: "Gradio test/eval UI", link: "/gradio-app" },
          ],
        },
        {
          text: "Extending",
          items: [
            { text: "Architecture", link: "/architecture" },
            { text: "Intent router", link: "/intent-router" },
            { text: "Workflows cookbook", link: "/workflows-cookbook" },
            { text: "Chat agent", link: "/agent" },
            { text: "Observability", link: "/observability" },
            { text: "Benchmarks", link: "/benchmarks" },
          ],
        },
      ],
      "/deep-dives/": [
        {
          text: "Strategy & history",
          items: [
            { text: "Proposal", link: "/deep-dives/proposal" },
            { text: "Phases roadmap", link: "/deep-dives/phases" },
            { text: "How we got here (journey)", link: "/deep-dives/journey" },
            { text: "Decisions (ADRs)", link: "/deep-dives/decisions" },
          ],
        },
        {
          text: "Reference",
          items: [
            { text: "How-to recipes", link: "/deep-dives/howto" },
            { text: "Changelog", link: "/deep-dives/changelog" },
          ],
        },
      ],
      "/": [
        {
          text: "Getting started",
          items: [
            { text: "Quickstart", link: "/guide/quickstart" },
            { text: "Installation", link: "/installation" },
            { text: "Operating", link: "/operating" },
          ],
        },
        {
          text: "Reference",
          items: [
            { text: "Architecture", link: "/architecture" },
            { text: "Intent router", link: "/intent-router" },
            { text: "API", link: "/api" },
            { text: "Workflows cookbook", link: "/workflows-cookbook" },
            { text: "Chat agent", link: "/agent" },
            { text: "Gradio app", link: "/gradio-app" },
            { text: "Observability", link: "/observability" },
            { text: "Benchmarks", link: "/benchmarks" },
          ],
        },
        {
          text: "Strategy & history",
          items: [
            { text: "Proposal", link: "/deep-dives/proposal" },
            { text: "Phases", link: "/deep-dives/phases" },
            { text: "Journey", link: "/deep-dives/journey" },
            { text: "Decisions", link: "/deep-dives/decisions" },
            { text: "Changelog", link: "/deep-dives/changelog" },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/protoLabsAI/protoBanana" },
    ],

    footer: {
      message: "Apache-2.0 licensed.",
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
