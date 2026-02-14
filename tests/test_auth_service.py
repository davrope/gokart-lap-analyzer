from __future__ import annotations

import unittest

from services.auth_service import AuthService


class _FakeAuth:
    def __init__(self) -> None:
        self.last_otp_payload = None
        self.last_verify_payload = None
        self.last_exchange_code = None
        self.last_oauth_payload = None
        self.session_set = None
        self.signed_out = False
        self.user_payload = {"id": "u1", "email": "driver@example.com"}

    def sign_in_with_otp(self, payload):
        self.last_otp_payload = payload

    def verify_otp(self, payload):
        self.last_verify_payload = payload

        class S:
            access_token = "at"
            refresh_token = "rt"

        class R:
            session = S()

        return R()

    def exchange_code_for_session(self, code):
        self.last_exchange_code = code.get("auth_code")

        class S:
            access_token = "at_code"
            refresh_token = "rt_code"

        class R:
            session = S()

        return R()

    def sign_in_with_oauth(self, payload):
        self.last_oauth_payload = payload
        return {"url": "https://accounts.google.com/o/oauth2/v2/auth?state=abc"}

    def set_session(self, access, refresh):
        self.session_set = (access, refresh)

    def get_user(self):
        class U:
            id = "u1"
            email = "driver@example.com"

        class R:
            user = U()

        return R()

    def sign_out(self):
        self.signed_out = True


class _FakeClient:
    def __init__(self):
        self.auth = _FakeAuth()


class AuthServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeClient()
        self.svc = AuthService(self.client, app_base_url="https://app.example.com")

    def test_send_magic_link(self) -> None:
        self.svc.send_magic_link("driver@example.com")
        p = self.client.auth.last_otp_payload
        self.assertEqual(p["email"], "driver@example.com")
        self.assertEqual(p["options"]["email_redirect_to"], "https://app.example.com")

    def test_consume_magic_link(self) -> None:
        session = self.svc.consume_magic_link({"token_hash": "abc", "type": "email"})
        self.assertIsNotNone(session)
        self.assertEqual(self.client.auth.last_verify_payload["token_hash"], "abc")
        self.assertEqual(self.client.auth.session_set, ("at", "rt"))

    def test_consume_magic_link_with_token_param_and_magiclink_type(self) -> None:
        session = self.svc.consume_magic_link({"token": "abc-token", "type": "magiclink"})
        self.assertIsNotNone(session)
        self.assertEqual(self.client.auth.last_verify_payload["token"], "abc-token")
        self.assertEqual(self.client.auth.last_verify_payload["type"], "email")
        self.assertEqual(self.client.auth.session_set, ("at", "rt"))

    def test_consume_magic_link_code_flow(self) -> None:
        session = self.svc.consume_magic_link({"code": "pkce-code"})
        self.assertIsNotNone(session)
        self.assertEqual(self.client.auth.last_exchange_code, "pkce-code")
        self.assertEqual(self.client.auth.session_set, ("at_code", "rt_code"))

    def test_get_google_auth_url(self) -> None:
        url = self.svc.get_google_auth_url()
        self.assertIn("accounts.google.com", url)
        self.assertEqual(self.client.auth.last_oauth_payload["provider"], "google")
        self.assertEqual(
            self.client.auth.last_oauth_payload["options"]["redirect_to"],
            "https://app.example.com",
        )

    def test_current_user(self) -> None:
        user = self.svc.current_user()
        self.assertIsNotNone(user)
        self.assertEqual(user.id, "u1")
        self.assertEqual(user.email, "driver@example.com")

    def test_sign_out(self) -> None:
        self.svc.sign_out()
        self.assertTrue(self.client.auth.signed_out)

    def test_persist_dict_session(self) -> None:
        session = self.svc._persist_and_dump_session({"access_token": "at2", "refresh_token": "rt2"})
        self.assertEqual(session["access_token"], "at2")
        self.assertEqual(session["refresh_token"], "rt2")
        self.assertEqual(self.client.auth.session_set, ("at2", "rt2"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
