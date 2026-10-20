package config

import (
	"fmt"

	"github.com/hack-fan/config"

	"github.com/crestalnetwork/intentkit/integrations/shared/alert"
)

type Config struct {
	Env     string `default:"local"`
	Debug   bool   `default:"false"`
	Release string `env:"RELEASE"`

	// DB
	DBHost     string `env:"DB_HOST"`
	DBPort     string `env:"DB_PORT" default:"5432"`
	DBName     string `env:"DB_NAME"`
	DBUsername string `env:"DB_USERNAME"`
	DBPassword string `env:"DB_PASSWORD"`

	// Internal API
	InternalBaseURL string `env:"INTERNAL_BASE_URL" default:"http://intent-api"`

	// Redis (used by the alert handler for shared rate limiting)
	RedisHost     string `env:"REDIS_HOST"`
	RedisPort     string `env:"REDIS_PORT" default:"6379"`
	RedisPassword string `env:"REDIS_PASSWORD"`
	RedisDB       int    `env:"REDIS_DB" default:"0"`

	// HTTP listen address for the public Lark event-subscription webhook +
	// OAuth callback. Lark pushes events here, so this must be reachable over
	// public HTTPS (put it behind the swarm's ingress / TLS terminator).
	ListenAddr string `env:"LARK_LISTEN_ADDR" default:":8084"`

	// ISV (store) app credentials. The SDK manages the app_access_token /
	// tenant_access_token chain from the app_ticket pushed to the webhook.
	LarkAppID             string `env:"LARK_APP_ID"`
	LarkAppSecret         string `env:"LARK_APP_SECRET"`
	LarkEncryptKey        string `env:"LARK_ENCRYPT_KEY"`        // AES key to decrypt events
	LarkVerificationToken string `env:"LARK_VERIFICATION_TOKEN"` // verifies events are from Lark
	LarkDomain            string `env:"LARK_DOMAIN" default:"feishu"`

	// Shared secret gating the internal-only reverse endpoints the Python API
	// calls on this service: /lark/exchange (ISV code->tenant_key) and
	// /lark/push (proactive send). The listen port is public (Lark posts events
	// to it), so these must present the secret; unset fails them closed.
	LarkInternalSecret string `env:"LARK_INTERNAL_SECRET"`

	// Alert (forwards Error+ slog records to Telegram/Slack)
	Alert alert.Config
}

func Load() (*Config, error) {
	var cfg Config
	if err := config.Load(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func (c *Config) DatabaseDSN() string {
	dsn := fmt.Sprintf("host=%s dbname=%s port=%s sslmode=disable TimeZone=UTC",
		c.DBHost, c.DBName, c.DBPort)
	if c.DBUsername != "" {
		dsn += fmt.Sprintf(" user=%s", c.DBUsername)
	}
	if c.DBPassword != "" {
		dsn += fmt.Sprintf(" password=%s", c.DBPassword)
	}
	return dsn
}
