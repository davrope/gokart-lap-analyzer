from __future__ import annotations

import unittest

from services.auth_service import AuthService


class _FakeAuth:
    def __init__(self) -> None:
        self.last_otp_payload = None
        self.last_verify_payload = None
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

    def test_current_user(self) -> None:
        user = self.svc.current_user()
        self.assertIsNotNone(user)
        self.assertEqual(user.id, "u1")
        self.assertEqual(user.email, "driver@example.com")

    def test_sign_out(self) -> None:
        self.svc.sign_out()
        self.assertTrue(self.client.auth.signed_out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
