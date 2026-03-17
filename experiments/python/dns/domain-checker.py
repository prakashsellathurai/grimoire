"""Check whether a domain looks registered or potentially available to buy.

This script combines two simple signals:
1. RDAP lookup to see whether the registry reports the domain as registered.
2. DNS resolution to see whether the domain currently has common records in use.

It is still only a heuristic. A registrar is the final source of truth for whether
you can actually buy a domain right now.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import dataclass
from typing import Iterable
from urllib import error, request


RDAP_URL_TEMPLATE = "https://rdap.org/domain/{domain}"
USER_AGENT = "domain-checker/1.0"


@dataclass()
class DomainCheckResult:
    domain: str
    rdap_status: str
    dns_status: str
    likely_available: bool
    registrar_hint: str | None
    notes: list[str]


def normalize_domain(domain: str) -> str:
    cleaned = domain.strip().lower()
    if cleaned.startswith(("http://", "https://")):
        cleaned = cleaned.split("://", 1)[1]
    cleaned = cleaned.split("/", 1)[0]
    cleaned = cleaned.removesuffix(".")
    # Add .com if no TLD is present
    if "." not in cleaned:
        cleaned += ".com"
    return cleaned


def check_rdap(domain: str, timeout: float) -> tuple[str, str | None, list[str]]:
    req = request.Request(
        RDAP_URL_TEMPLATE.format(domain=domain),
        headers={"User-Agent": USER_AGENT, "Accept": "application/rdap+json, application/json"},
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = json.load(response)
    except error.HTTPError as exc:
        if exc.code == 404:
            return "not found", None, ["Registry RDAP returned 404."]
        return "error", None, [f"RDAP HTTP error: {exc.code}."]
    except error.URLError as exc:
        return "error", None, [f"RDAP network error: {exc.reason}."]
    except TimeoutError:
        return "error", None, ["RDAP request timed out."]
    except json.JSONDecodeError:
        return "error", None, ["RDAP response was not valid JSON."]

    registrar_hint = None
    notes: list[str] = []

    entities = payload.get("entities", [])
    for entity in entities:
        roles = entity.get("roles", [])
        if "registrar" not in roles:
            continue
        vcard_array = entity.get("vcardArray", [])
        if len(vcard_array) < 2:
            continue
        for card in vcard_array[1]:
            if len(card) >= 4 and card[0] == "fn":
                registrar_hint = str(card[3])
                break
        if registrar_hint:
            break

    status_values = payload.get("status") or []
    if status_values:
        notes.append(f"RDAP status: {', '.join(map(str, status_values))}.")

    ldh_name = payload.get("ldhName")
    if ldh_name:
        notes.append(f"Registry returned record for {ldh_name}.")

    return "registered", registrar_hint, notes


def check_dns(domain: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    record_types = ("80", "443")

    try:
        answers = socket.getaddrinfo(domain, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return "not resolving", [f"DNS resolution failed: {exc}."]

    addresses = sorted({entry[4][0] for entry in answers})
    if addresses:
        notes.append(f"Resolved addresses: {', '.join(addresses[:4])}.")

    # Try common subdomain patterns as a soft signal for live configuration.
    for port in record_types:
        try:
            with socket.create_connection((domain, int(port)), timeout=1.5):
                notes.append(f"Accepted TCP connection on port {port}.")
                return "active", notes
        except OSError:
            continue

    return "resolving", notes


def evaluate_domain(domain: str, timeout: float) -> DomainCheckResult:
    notes: list[str] = []
    rdap_status, registrar_hint, rdap_notes = check_rdap(domain, timeout)
    dns_status, dns_notes = check_dns(domain)
    notes.extend(rdap_notes)
    notes.extend(dns_notes)

    likely_available = rdap_status == "not found" and dns_status == "not resolving"
    if likely_available:
        notes.append("No registry record and no DNS resolution were found.")
    elif rdap_status == "registered":
        notes.append("Registry data indicates the domain is already registered.")
    else:
        notes.append("Availability is uncertain; verify with a registrar before buying.")

    return DomainCheckResult(
        domain=domain,
        rdap_status=rdap_status,
        dns_status=dns_status,
        likely_available=likely_available,
        registrar_hint=registrar_hint,
        notes=notes,
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a domain looks registered or potentially available to buy.",
    )
    parser.add_argument("domains", nargs="+", help="One or more domains to inspect.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Network timeout in seconds for the RDAP request. Default: 5.0",
    )
    return parser.parse_args(list(argv))


def print_result(result: DomainCheckResult) -> None:
    verdict = "POSSIBLY AVAILABLE" if result.likely_available else "LIKELY REGISTERED / UNCERTAIN"
    print(f"\nDomain: {result.domain}")
    print(f"Verdict: {verdict}")
    print(f"RDAP: {result.rdap_status}")
    print(f"DNS: {result.dns_status}")
    if result.registrar_hint:
        print(f"Registrar: {result.registrar_hint}")
    print("Notes:")
    for note in result.notes:
        print(f"  - {note}")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    for raw_domain in args.domains:
        domain = normalize_domain(raw_domain)
        if not domain:
            print(f"\nSkipping invalid input: {raw_domain!r}")
            continue
        print_result(evaluate_domain(domain, args.timeout))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
