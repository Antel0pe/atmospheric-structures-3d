import path from "node:path";

const DEFAULT_DATA_DIR = "public";

export function getDataRootPath() {
  const configured = process.env.DATA_DIR?.trim() || DEFAULT_DATA_DIR;

  if (path.isAbsolute(configured)) return configured;
  return path.join(process.cwd(), configured);
}
