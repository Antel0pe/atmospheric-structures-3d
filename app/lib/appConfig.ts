import deployedAppConfig from "../../config/app-config.deployed.json";
import devAppConfig from "../../config/app-config.dev.json";

export type AppMode = "dev" | "deployed";

export type SliderDateRange = {
  startDate: string;
  endDate: string;
};

export type AppConfig = {
  dataDir: string;
  sliderDateRange: SliderDateRange;
};

const APP_MODE: AppMode =
  process.env.NEXT_PUBLIC_APP_MODE?.trim() === "dev" ? "dev" : "deployed";

const APP_CONFIG_BY_MODE = {
  dev: devAppConfig,
  deployed: deployedAppConfig,
} satisfies Record<AppMode, AppConfig>;

export function getAppMode(): AppMode {
  return APP_MODE;
}

export function isDevMode(): boolean {
  return APP_MODE === "dev";
}

export function getAppConfig(): AppConfig {
  return APP_CONFIG_BY_MODE[APP_MODE];
}
