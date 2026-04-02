import path from "node:path";

import { getAppConfig } from "../../lib/appConfig";

export function getFilesystemDataRootPath() {
  const configured = getAppConfig().dataDir;

  if (!configured || !path.isAbsolute(configured)) {
    return null;
  }

  return configured;
}
