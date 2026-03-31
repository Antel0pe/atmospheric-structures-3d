import path from "node:path";

export function getFilesystemDataRootPath() {
  const configured = process.env.DATA_DIR?.trim();

  if (!configured || !path.isAbsolute(configured)) {
    return null;
  }

  return configured;
}
