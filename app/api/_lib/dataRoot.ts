import path from "node:path";

import { getAppConfig } from "../../lib/appConfig";

export function getDataRootPath() {
  const configured = getAppConfig().dataDir;

  if (path.isAbsolute(configured)) return configured;
  return path.join(process.cwd(), configured);
}
