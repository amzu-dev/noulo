import pytest

from noulo.config import ConfigError, Settings


def make(**env) -> Settings:
    return Settings(_env_file=None, **env)


def test_defaults_are_loopback_and_port_8787():
    s = make()
    assert (s.host, s.port) == ("127.0.0.1", 8787)
    assert s.base_url == "http://127.0.0.1:8787"


def test_reads_prefixed_environment_variables(monkeypatch):
    monkeypatch.setenv("NOULO_PORT", "9999")
    monkeypatch.setenv("NOULO_LEARNING_ENABLED", "false")
    s = Settings(_env_file=None)
    assert s.port == 9999 and s.learning_enabled is False


def test_reads_dotenv_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("NOULO_MODEL=nli-mobilebert-int8\nNOULO_MAX_CONCURRENCY=3\n")
    s = Settings(_env_file=env)
    assert s.model == "nli-mobilebert-int8" and s.max_concurrency == 3


def test_non_loopback_host_requires_explicit_network_opt_in():
    with pytest.raises(ConfigError, match="NOULO_ALLOW_NETWORK"):
        make(host="0.0.0.0").check()
    make(host="0.0.0.0", allow_network=True).check()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_loopback_hosts_need_no_opt_in(host):
    make(host=host).check()


def test_cors_defaults_to_localhost_only_never_wildcard():
    s = make()
    assert s.cors_enabled
    assert "*" not in s.cors_origin_list
    assert s.cors_origin_regex is not None


def test_cors_origins_parse_comma_separated_list():
    s = make(cors_origins="https://a.example, https://b.example")
    assert s.cors_origin_list == ["https://a.example", "https://b.example"]


def test_wildcard_cors_is_rejected():
    with pytest.raises(ConfigError, match="CORS"):
        make(cors_origins="*").check()


def test_limits_are_exposed_as_validation_limits():
    s = make(max_input_chars=10, max_choices=4)
    assert s.limits.max_input_chars == 10 and s.limits.max_choices == 4


def test_api_key_is_hidden_in_repr():
    s = make(api_key="super-secret")
    assert "super-secret" not in repr(s)
    assert s.api_key.get_secret_value() == "super-secret"


def test_public_dict_redacts_secrets():
    s = make(api_key="super-secret", openai_api_key="sk-live")
    public = s.public_dict()
    assert public["api_key"] == "***" and public["openai_api_key"] == "***"
    assert "super-secret" not in str(public) and "sk-live" not in str(public)


def test_every_setting_has_help_text():
    from noulo.config import SETTING_HELP

    assert set(SETTING_HELP) == set(Settings.model_fields)
    assert all(text and text[0].isupper() for text in SETTING_HELP.values())


def test_secret_settings_are_flagged():
    from noulo.config import SECRET_FIELDS

    assert {"api_key", "openai_api_key", "memory_api_key"} <= set(SECRET_FIELDS)
