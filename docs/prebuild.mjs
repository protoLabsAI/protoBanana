// Copy root-level markdown into docs/deep-dives/ before each build.
// Single source of truth stays at the repo root (where contributors edit);
// VitePress sees fresh copies under deep-dives/.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");
const target = join(__dirname, "deep-dives");

mkdirSync(target, { recursive: true });

const files = ["PROPOSAL.md", "PHASES.md", "JOURNEY.md", "HOWTO.md", "DECISIONS.md", "CHANGELOG.md"];
for (const f of files) {
  copyFileSync(join(root, f), join(target, f.toLowerCase()));
  console.log(`copied ${f} -> deep-dives/${f.toLowerCase()}`);
}
