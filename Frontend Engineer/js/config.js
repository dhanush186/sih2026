// Local development default. In production this must point at an
// https:// origin (checklist #19 Force HTTPS) — never ship an http://
// API endpoint. Update the CSP "connect-src" in each HTML file's
// <head> to match whenever this value changes.
const API_BASE_URL = "http://localhost:8000";
