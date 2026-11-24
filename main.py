from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import base64
import json
import hmac
import re
import hashlib
import html as html_lib
import httpx  # already installed as part of the openai package
from typing import Optional
from datetime import datetime, timezone

# PostgreSQL — usage counts are stored in the database so they survive restarts
import psycopg
from psycopg_pool import ConnectionPool

app = FastAPI(title="Half-Pint Resale API")

# CORS — allow the PWA and any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client — reads OPENAI_API_KEY from environment automatically
openai_client = OpenAI()

# --- Database setup ---
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Railway sometimes provides the older "postgres://" scheme; psycopg wants "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

db_pool = None
if DATABASE_URL:
    try:
        db_pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=True)
        print("✓ Database pool created")
    except Exception as e:
        print(f"!! Database pool error: {e}")
else:
    print("!! DATABASE_URL not set — usage counts will NOT persist")


def init_database():
    """Create the tables that track code usage. Safe to run on every startup."""
    if not db_pool:
        return
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            # One row per access code, holding how many uses have been consumed.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS code_usage (
                    code TEXT PRIMARY KEY,
                    used INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # One row per item that has already been charged a use.
            # This is what makes edits free and retries safe: charging the same
            # item_id twice is impossible.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS consumed_items (
                    code TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (code, item_id)
                )
            """)
        conn.commit()
    print("✓ Database tables ready")


def get_used(code: str) -> int:
    """How many uses this code has consumed. Falls back to 0 if no DB."""
    if not db_pool:
        return 0
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT used FROM code_usage WHERE code = %s", (code,))
            row = cur.fetchone()
            return row[0] if row else 0


def consume_use(code: str, item_id: Optional[str] = None) -> int:
    """
    Deduct one use from a code and return the new used total.

    If item_id is supplied and that item has already been charged, nothing is
    deducted — this is what makes editing an item free and makes a retried
    network request safe.
    """
    if not db_pool:
        return 0
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            if item_id:
                # Claim this item. If the row already exists, it was charged before.
                cur.execute(
                    """
                    INSERT INTO consumed_items (code, item_id)
                    VALUES (%s, %s)
                    ON CONFLICT (code, item_id) DO NOTHING
                    RETURNING item_id
                    """,
                    (code, item_id),
                )
                if cur.fetchone() is None:
                    # Already charged — return the current count unchanged.
                    cur.execute("SELECT used FROM code_usage WHERE code = %s", (code,))
                    row = cur.fetchone()
                    conn.commit()
                    return row[0] if row else 0

            cur.execute(
                """
                INSERT INTO code_usage (code, used, updated_at)
                VALUES (%s, 1, NOW())
                ON CONFLICT (code) DO UPDATE
                    SET used = code_usage.used + 1, updated_at = NOW()
                RETURNING used
                """,
                (code,),
            )
            new_used = cur.fetchone()[0]
        conn.commit()
    return new_used


# --- Free period configuration ---
# While a free window is active, tagging is open to everyone: no code needed and
# nothing is deducted. Tags created during the window are permanently free — they
# are never charged retroactively when the window closes.
#
# Set these as Railway environment variables (UTC ISO dates, YYYY-MM-DD):
#   FREE_PERIOD_START=2026-08-01
#   FREE_PERIOD_END=2026-09-15
# Leave either unset to disable the free period entirely.
#
# The END date is inclusive: an end of 2026-09-15 stays free through the whole
# of September 15 and starts charging on September 16.
FREE_PERIOD_START = os.environ.get("FREE_PERIOD_START", "").strip()
FREE_PERIOD_END = os.environ.get("FREE_PERIOD_END", "").strip()

# AI photo analysis costs real money on every call (OpenAI + Google Vision),
# unlike typed and dictated tagging which cost nothing per item. So the free
# period covers AI photos ONLY if you opt in:
#   FREE_PERIOD_INCLUDES_AI=true
# Default is false: during a free window, typed and dictated tagging is open to
# all, but AI photo auto-fill still requires a code and still deducts a use.
FREE_PERIOD_INCLUDES_AI = os.environ.get("FREE_PERIOD_INCLUDES_AI", "false").strip().lower() in ("1", "true", "yes")

# --- Entry method pricing ---
# The app offers three ways to tag an item:
#   Manual Entry       typing only. ALWAYS free, no code, no deduction. The app
#                      does not even call /consume for these.
#   Dictation Tagging  speak the fields. Free when DICTATION_FREE is on,
#                      otherwise needs a code and deducts one use per item.
#   AI Auto-Fill       two photos. Always needs a code and deducts one use,
#                      unless a free period has FREE_PERIOD_INCLUDES_AI on.
#
# DICTATION_FREE is a standing switch, independent of the free period, so
# dictation can be opened up without also opening up AI photos:
#   DICTATION_FREE=true
# Default is false. Flipping it to true makes dictation completely open — no
# code required and nothing deducted, for everyone.
DICTATION_FREE = os.environ.get("DICTATION_FREE", "false").strip().lower() in ("1", "true", "yes")


def _parse_date(value: str):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def free_period_status():
    """Return the current free-period state, evaluated on the server clock."""
    start = _parse_date(FREE_PERIOD_START)
    end = _parse_date(FREE_PERIOD_END)

    if not start or not end:
        return {"free_now": False, "start": None, "end": None, "configured": False}

    today = datetime.now(timezone.utc).date()
    return {
        "free_now": start <= today <= end,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "configured": True,
    }


# Google Vision setup — reads GOOGLE_CREDENTIALS_JSON env var (base64-encoded JSON)
def get_vision_client():
    try:
        from google.cloud import vision
        from google.oauth2 import service_account

        creds_b64 = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not creds_b64:
            raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set")

        creds_json = base64.b64decode(creds_b64).decode("utf-8")
        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        return vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        print(f"Vision client init error: {e}")
        return None

# Access code database - hardcoded for simplicity
# Format: code -> { "total_uses": int, "used": int, "type": "trial"|"season" }
def initialize_codes():
    """Initialize all access codes from hardcoded data."""
    codes = {}
    
    # Trial codes (400 codes, 10 uses each)
    trial_codes = [
        "TR-VBT4DYY7", "TR-QEPC5VKG", "TR-V67MHVTT", "TR-SZ5QMBDE", "TR-YPF5B78S",
        "TR-YW6X8CLR", "TR-QJ89PWD7", "TR-59TV7UV5", "TR-3ERXY5S6", "TR-YFUEQJP8",
        "TR-V3VJSC7Y", "TR-N86NLASY", "TR-UA6TFHFY", "TR-VAUFY7TK", "TR-3RS9G2FJ",
        "TR-4KQEZR9W", "TR-YE6NN8H8", "TR-BWQ6KZNM", "TR-PJVNWCVT", "TR-JXAAVKH2",
        "TR-DDP9K2MR", "TR-PVGDVSQ4", "TR-KVSFSVTF", "TR-D2UXHAH9", "TR-B9N8QM99",
        "TR-PGX3AC8E", "TR-A89WVA3L", "TR-T5TS3Y76", "TR-EPLPNJ2M", "TR-FD5PZAPV",
        "TR-Q9TB27YV", "TR-XX2H6HZJ", "TR-J7YC3Z32", "TR-3XVQCNJF", "TR-LDSNSXM9",
        "TR-35CYDHLP", "TR-EFQAXJSD", "TR-RAQAU2E3", "TR-JBDLF7NM", "TR-GM9W7LDY",
        "TR-B3V3G26A", "TR-JZ2D8BQX", "TR-9QSVSX4J", "TR-M5L2NTTA", "TR-Q6WWHQ68",
        "TR-VHM9DQSE", "TR-8F8PQZBK", "TR-K3SRCVG2", "TR-4ZMRXXYM", "TR-CTWBSDHL",
        "TR-TVHFT2QP", "TR-UKTFE4FF", "TR-QFZ4UNNW", "TR-HQ7EBBRF", "TR-NS6MQ32L",
        "TR-GP446PPH", "TR-CN83SRWU", "TR-G7289C5Y", "TR-VJ7ZKA2S", "TR-9B8JJGWY",
        "TR-LKCVJ4TW", "TR-6EPWQ5BV", "TR-QAXD9VWN", "TR-SYFZBBU8", "TR-3CV47CB5",
        "TR-LE7RAYCA", "TR-3NNKEB7L", "TR-ZJJAYMYG", "TR-JUTJ5MXY", "TR-USGB5JMU",
        "TR-TJGYPGDU", "TR-JAF7HZ9G", "TR-8LBJAAPB", "TR-L9DMQNY5", "TR-C3T8XGRN",
        "TR-A3U49H32", "TR-6834H8HW", "TR-EMGVREH5", "TR-HMKA3DR3", "TR-W5CDEAXK",
        "TR-TA285P99", "TR-7EAWAKPT", "TR-8TV5QB67", "TR-TE2DMJCG", "TR-LT3ZAQ2V",
        "TR-MC7LC7S3", "TR-P8VKF2D3", "TR-SC62TV3H", "TR-4L9GUAY5", "TR-H9WFW9ZQ",
        "TR-GPSWR34Z", "TR-VTRM36E8", "TR-ATUD4XCA", "TR-FHLH9YU8", "TR-Y7GFCM4H",
        "TR-NYELD3NW", "TR-65WY6SMP", "TR-VHCDRX84", "TR-SEKBNAYB", "TR-DWSKKFDP",
        "TR-3W7DAD7P", "TR-ME434SSU", "TR-KULYJZTU", "TR-U3ZZJMCP", "TR-M825PP7M",
        "TR-WYZ8H58T", "TR-NC2VGHWZ", "TR-ZCAAB4LY", "TR-V2JVDMWA", "TR-25UJCVSX",
        "TR-8HUFRWQT", "TR-SWESS79L", "TR-2VZKU29A", "TR-3QJVKRSN", "TR-QFJPXWEV",
        "TR-95WX5GKL", "TR-LQSU4NT7", "TR-6Q3S6BKR", "TR-HXKMCG78", "TR-XFEPNPJF",
        "TR-S2ZUXV4D", "TR-BMWD7KNF", "TR-69M8J9QM", "TR-QJGH7JKD", "TR-MV4T9UMH",
        "TR-5J8A88Q7", "TR-HBHZ3ECY", "TR-5ZTYFX6N", "TR-7AJWX85L", "TR-YHBNXLFL",
        "TR-NMY8LKVT", "TR-DNRW3Q3T", "TR-MJEW4G6A", "TR-43HXSM3K", "TR-MRSRFLSD",
        "TR-E2ANCDEJ", "TR-ANG2NYZV", "TR-6W2843MC", "TR-UN8KZA5J", "TR-8UD6G7RF",
        "TR-TQHLCUJV", "TR-LVLT5JPH", "TR-ER6S844G", "TR-UKFCU8R6", "TR-EXYYMRNY",
        "TR-EFC6JAAW", "TR-27LQNKVT", "TR-B7BDKBQU", "TR-KZU7D57U", "TR-LDFCELWN",
        "TR-KPDVSHNK", "TR-TUVJRW3B", "TR-JJSRWL8S", "TR-NWLQ89KT", "TR-79QU7L22",
        "TR-J3TRT9DC", "TR-T2Q52BT3", "TR-NQ33N76H", "TR-MYDD5KNK", "TR-B67DB2FP",
        "TR-SM5DN42R", "TR-APVMKXRW", "TR-5ZJNZBYX", "TR-H9ENRJML", "TR-QWH82FJ6",
        "TR-5XA9FHQS", "TR-GXNG7EVJ", "TR-4VKTCJB9", "TR-2ZCATXC9", "TR-ARRJYLFC",
        "TR-KLNFD8WT", "TR-FQFX6HKL", "TR-2QAVJCBG", "TR-JD3LPJCH", "TR-E9ZKXGL7",
        "TR-GNBVR5WL", "TR-QJFC5FKJ", "TR-TQ2FPF9H", "TR-YGDZJ9R2", "TR-DH7QLM72",
        "TR-54ZTBREA", "TR-L9HB8QTL", "TR-4BNU3NCG", "TR-9SH55627", "TR-EYLGN8R9",
        "TR-PDRZ5BLE", "TR-TGZ9MQYR", "TR-VASUPM98", "TR-ADUVFKPB", "TR-CAUUEGYY",
        "TR-YRERE236", "TR-UQ5XW4GT", "TR-GPX62GXG", "TR-QDRBMPPJ", "TR-D3MUBSC2",
        "TR-GFKYFLWQ", "TR-4RM826E4", "TR-3X62GZXX", "TR-EE6ZMMMJ", "TR-KNXCUVVT",
        "TR-5WAF7HY9", "TR-MGWXH8GF", "TR-UAFDY6GP", "TR-CYWN2TFA", "TR-CRRVVYRQ",
        "TR-PX64K6JU", "TR-BSTCG8HP", "TR-9FQDS5SZ", "TR-HEEA4RDY", "TR-X7QHD9AN",
        "TR-MKBQN4BL", "TR-Y3G87G8Q", "TR-MTB27MP6", "TR-DHN9VNHY", "TR-HDW49ADK",
        "TR-9JGLSPTX", "TR-P6XZGS2T", "TR-AB2YQWAG", "TR-XET6GUGK", "TR-59FB4CAB",
        "TR-ZRUYZAEM", "TR-ENXVZXVK", "TR-4MCH8CSZ", "TR-THFKMRY8", "TR-46855X75",
        "TR-FVDDWY9Q", "TR-CB3EQ3L3", "TR-QARUGKEC", "TR-5MCAK98D", "TR-FTT9C5NK",
        "TR-RT5WT72V", "TR-XHE7N88H", "TR-V567Y5LM", "TR-RRFV6BRQ", "TR-CLAVAVFR",
        "TR-EQ24FG85", "TR-RTZBFWC7", "TR-8QYBN55D", "TR-B6WB7TGD", "TR-WDEYP2DG",
        "TR-XS4FYPU3", "TR-X88NGX7X", "TR-4QV8SK3Y", "TR-YGF7HC2S", "TR-Q3H88UW4",
        "TR-FQFQ538U", "TR-YVZJ7EQW", "TR-JL7CFRDP", "TR-JF55UV4M", "TR-A37J8HWC",
        "TR-925KRMCM", "TR-A9ELT5VJ", "TR-VS8GGU4M", "TR-T24TXH9T", "TR-C4BLCK9J",
        "TR-PQWWZQBQ", "TR-QHDDZKX5", "TR-QNCXWL9L", "TR-U963GUZW", "TR-STKVCFCH",
        "TR-AMTYSVXH", "TR-WF5EUMDZ", "TR-BZGB8Y4Z", "TR-XXH2PFFX", "TR-US4B4PZ8",
        "TR-AB6KQVUD", "TR-FZRVFRPL", "TR-ESQP6VNC", "TR-R7JVWHCD", "TR-PJE2WQ5B",
        "TR-US7QRZRE", "TR-ZYNALMYS", "TR-2YF48RJU", "TR-VQFTWVQY", "TR-6AJ5UX5A",
        "TR-9LALZ4EV", "TR-W4SX3GKS", "TR-3HNQG6HW", "TR-D5J8QE6F", "TR-2EYSJAGE",
        "TR-AT6HAD6G", "TR-TTFMJB9B", "TR-4F95H3WJ", "TR-45ZGDH5A", "TR-BNACCD9K",
        "TR-RG6AQ6A5", "TR-CGKBLKWY", "TR-8F6B5PN2", "TR-4QK3PPCE", "TR-NJ7HNYGC",
        "TR-MVL8TXZB", "TR-FL6PXCBR", "TR-EBPN6JE3", "TR-DD5SUMSY", "TR-4ZHFEGS5",
        "TR-JT96TARU", "TR-B7X8CJT8", "TR-R47RFV8F", "TR-K8NUXRJP", "TR-9AJJZSQP",
        "TR-YYCTN7NL", "TR-4PRUWL6U", "TR-Y7DR62F6", "TR-PJ4VTM2X", "TR-HY89AF4Y",
        "TR-X5RSWML8", "TR-D3WG8VGX", "TR-DZJJ8XTB", "TR-72GA99DM", "TR-EUE7KMXJ",
        "TR-TTGUS2EV", "TR-B3T6QD3B", "TR-X5P8R434", "TR-TS5UYJJV", "TR-MJ9AP4N2",
        "TR-YJGKCBGY", "TR-GGVZ9WVW", "TR-NTFGQPN6", "TR-DG6GCR4G", "TR-SCWXNEQ8",
        "TR-5CQNNLRH", "TR-C5483HVV", "TR-JYSTRADY", "TR-6MMV9GTH", "TR-TV6RZWQX",
        "TR-RHAMGZVX", "TR-3QWGHNWT", "TR-EMLBXDAK", "TR-CK2DQQMQ", "TR-6KY3PXS2",
        "TR-9VUSUT6N", "TR-RQQ4P7WT", "TR-45QV9KXM", "TR-3DHDNZ9S", "TR-RFVK2RVF",
        "TR-FXDU4XQY", "TR-RV3QSSYC", "TR-3NBNFWPU", "TR-7GHL9RXK", "TR-UF347VBL",
        "TR-T7P5737S", "TR-YMSBJE2A", "TR-NMUPFRRB", "TR-4C9MPZ8Y", "TR-FD7HYJP2",
        "TR-S4H3NACK", "TR-F3ZWTTUC", "TR-HZZYZJUR", "TR-ZB35YF9R", "TR-CEK7GVLM",
        "TR-VWGZPD6W", "TR-N2X4YNE8", "TR-XLVCPQT2", "TR-PHFWHBS6", "TR-TEMBAZSY",
        "TR-8FG3C3HM", "TR-Y7A6L3U5", "TR-9MB44K2L", "TR-LZX5N9VD", "TR-Q2L9AW2V",
        "TR-TYAYGUGJ", "TR-KQ2KQKGT", "TR-WU6QAT3K", "TR-GNL7ZTMH", "TR-UBQUNLX6",
        "TR-LLD56WP8", "TR-DEDJP8LB", "TR-KJ6EKVPA", "TR-8W8RDSCR", "TR-UNPCPJMS",
        "TR-5M3HHRR9", "TR-HSKDMLG9", "TR-R4DKUGAG", "TR-9EFP9LDP", "TR-ZTBRRFSK",
        "TR-Y93YYNLQ", "TR-CLARLVND", "TR-PNJBH2BV", "TR-EPWHZNR8", "TR-2R5UPFDW",
        "TR-4QHBQJED", "TR-RG6QSSNS", "TR-4UCWK46B", "TR-7D98SP8X", "TR-3A3NMFLD",
        "TR-XXJQFBP3", "TR-V9Q35FYV", "TR-5RRF4NDJ", "TR-5HEAMYRD", "TR-DQTEE3ZD",
        "TR-ESGMSJY9", "TR-HPU6Q8SH", "TR-F96KNJSV", "TR-F8QDP9WY", "TR-AB6BNMTS",
        "TR-Q86LK458", "TR-WVQJY2MD", "TR-4V5T9SJW", "TR-3H8J3H2V", "TR-Q9HDP4WY"
    ]
    
    for code in trial_codes:
        codes[code] = {"total_uses": 10, "used": 0, "type": "trial"}
    
    # Season codes (400 codes, 200 uses each)
    season_codes = [
        "SS-J9JNSZDT", "SS-KXXV2HK2", "SS-BAASWGWJ", "SS-G879QB6M", "SS-7J6LJMEW",
        "SS-5TB6KFCZ", "SS-PK8CCWH7", "SS-NW7LFAL4", "SS-PWW2NWLQ", "SS-PD643L2H",
        "SS-PM6V5RTW", "SS-JXNAJ5VA", "SS-FCQBLWPE", "SS-C3T3VEK2", "SS-RRNPJSAS",
        "SS-W2SFW83U", "SS-4AF5PYUC", "SS-V7WPUP3H", "SS-SPUCKYEE", "SS-QNJCW7WY",
        "SS-P9BW4VKD", "SS-3TX53Q7K", "SS-KU93MVN7", "SS-XRPXZWXR", "SS-2WWQCQP7",
        "SS-YAWZKHMA", "SS-RX5QUA8U", "SS-TF6YUU6P", "SS-5YK6UVA6", "SS-4A52KN9A",
        "SS-X9L4GD9T", "SS-AUQKHJF4", "SS-GRUZMNUA", "SS-T8K4TFVW", "SS-S2UQSDXU",
        "SS-UUCXDDCB", "SS-W78J6MM9", "SS-MV9WH87J", "SS-83NCVXK9", "SS-MLF2VVRT",
        "SS-LNPC9Y2E", "SS-8MTCLL8C", "SS-6F4BQ9EY", "SS-Q8YZLHCE", "SS-66GX474N",
        "SS-3Q7FKPWX", "SS-E2SQCDVC", "SS-5A7H6YQA", "SS-L6ZTFL6W", "SS-JN85WT4V",
        "SS-B37P3QH4", "SS-KJVGAXZ8", "SS-LFHQLULS", "SS-8BNQF2HN", "SS-C2WSC42G",
        "SS-KU7MKUCJ", "SS-LSJB9MTG", "SS-S7Z3YLYP", "SS-9XCNAMSX", "SS-VBH4YLAN",
        "SS-YU7YNK47", "SS-RLJNUR3B", "SS-4W8CYSB2", "SS-DEQHMKG4", "SS-YQ5TSCB6",
        "SS-ARJ9X6LY", "SS-8C5C8U9C", "SS-75EEGR2T", "SS-7U8546UB", "SS-Z9RFSBZS",
        "SS-Y2492WCB", "SS-WGR8G87V", "SS-F5F8QZEB", "SS-58ZYTLWN", "SS-RH5WWFPU",
        "SS-FBR33FZ8", "SS-QHSJ2ASR", "SS-RGJSLWJC", "SS-NHAMZZGJ", "SS-EDTUX97V",
        "SS-HY7KZR4U", "SS-RP69ZRHQ", "SS-NNBBWHYA", "SS-WV26U83L", "SS-QDSGNPZU",
        "SS-FDVGYWVG", "SS-FSF4ADL9", "SS-KNXN9GPS", "SS-M9FW956G", "SS-QXJ76ZEW",
        "SS-LBS76FVS", "SS-Q5W9995V", "SS-F2DQW956", "SS-BH8UXVT5", "SS-4PBGTVK8",
        "SS-WMH56MLZ", "SS-BDTUPG6C", "SS-3966BVQ3", "SS-N3WARYQV", "SS-6HHALHJ4",
        "SS-CZGSNX2V", "SS-FUP8CYRR", "SS-KMSTYNNJ", "SS-VY66GLVF", "SS-AH9SUNLY",
        "SS-MXFK25JC", "SS-JP7ARFQJ", "SS-AJ8HZS72", "SS-SPL3ZPQ5", "SS-AMZ4UKLE",
        "SS-53J95AXK", "SS-GEUAVTFP", "SS-DPLBHRTC", "SS-32NQ2X4L", "SS-VTFM3GZL",
        "SS-JNLDWZ27", "SS-XJJQ3G9U", "SS-LCY4DBSE", "SS-X6P6EGPJ", "SS-3MHUUUET",
        "SS-887PJY47", "SS-WBN5DDS9", "SS-FAHY98JW", "SS-G7QZR6PE", "SS-RHZPB3MD",
        "SS-2J4V2Y8P", "SS-SM48FBQP", "SS-MAWUZG67", "SS-5JK44CYU", "SS-SY53R9HD",
        "SS-NNG6NCNW", "SS-N55CF4ZN", "SS-PTM47SDX", "SS-Q5NC6N7B", "SS-HPNQCPNE",
        "SS-DXUM2HJD", "SS-8XA3MCAB", "SS-HZ5MRGH8", "SS-372758FR", "SS-RPYVZCLC",
        "SS-XDVXYUGA", "SS-AGUTUA9X", "SS-794EHWG4", "SS-U6AY9HRL", "SS-46ZW2VQU",
        "SS-N5JWPVPH", "SS-VTZ7E535", "SS-KY6X5S5P", "SS-N6Y5ZXXX", "SS-FKHHU58N",
        "SS-YLX5K9QL", "SS-RFR5YJR7", "SS-LQC7VVMD", "SS-QQMYVF7X", "SS-B22M86R9",
        "SS-U7A8X2H8", "SS-939KMSQW", "SS-TD4HQM3X", "SS-RUKNYRWY", "SS-CADW8D8X",
        "SS-6HB8HXD7", "SS-HFDWYSHY", "SS-L6QGA8EF", "SS-U4R9E4WZ", "SS-A7LMVWFX",
        "SS-ND9WTJ4Y", "SS-NML3LQQD", "SS-8AYMUGBB", "SS-5RYJPZBY", "SS-MXF7QRXA",
        "SS-4UFYP3TS", "SS-3W5R5HVE", "SS-XVENZPC3", "SS-6QPU4RMJ", "SS-S2JV7L4R",
        "SS-TCWQHTVN", "SS-TC4TZ36H", "SS-C77UHCPH", "SS-2K2WUYFS", "SS-R9Z4RF2F",
        "SS-AFG8FDWF", "SS-DCWQ8FYG", "SS-CD2WPN3G", "SS-D46X269Y", "SS-Y58FD3Y3",
        "SS-C7RUUSBH", "SS-P62Z7USD", "SS-M2AU2LFR", "SS-FZBHKZW3", "SS-RHA57Y32",
        "SS-3LYHASE8", "SS-LL5YU6YR", "SS-C56SRSZQ", "SS-82DCW4EM", "SS-YDBH8SDW",
        "SS-E9AUTPER", "SS-48T3LJSC", "SS-2HMQ76TD", "SS-5TXGMLXY", "SS-J3GBEF9S",
        "SS-JNVQFRSR", "SS-FD4XYCCC", "SS-TVGMVSXL", "SS-VGF5SYHH", "SS-FJ8R9QMG",
        "SS-2NRZVLKM", "SS-CDL43CD8", "SS-TH5U46A4", "SS-MXB87TV9", "SS-ALUPEGFT",
        "SS-ZJYFP3W2", "SS-EGEEUQE2", "SS-AQKXLE7G", "SS-3FZPS9RD", "SS-CP5DBMMQ",
        "SS-R6ZAD2CQ", "SS-23VSPP58", "SS-X2CZB2JK", "SS-T34EJC2Z", "SS-TYLPYN8T",
        "SS-67JB4LLR", "SS-PMHJCKPW", "SS-7LKDPHJ2", "SS-U8QJGZBU", "SS-EHRGFBGP",
        "SS-AZLQ3T2B", "SS-RQJ5FJYG", "SS-BFJ264EJ", "SS-6R92X78H", "SS-BELLFGMH",
        "SS-M2QXPTYL", "SS-R6NLENPN", "SS-4523P25G", "SS-P7M6E9SC", "SS-VNV6YHHM",
        "SS-3FTKN3F4", "SS-XPJE5UP6", "SS-LAS2DCWF", "SS-BP87GT5N", "SS-37XPL24J",
        "SS-WHR9H2NC", "SS-W8R7YHM8", "SS-5XN7V2R8", "SS-8XQW8EMB", "SS-PQ5YZHHU",
        "SS-BLFHQZK4", "SS-H9C9FX9B", "SS-VKXXJZZM", "SS-W8P7ANH6", "SS-H4DXFSYV",
        "SS-2QWFAHQW", "SS-S7WWJH4Q", "SS-BE24AGYC", "SS-88X3WUTR", "SS-GYREJS46",
        "SS-GXZMUZVA", "SS-MB3AQVK2", "SS-MG3GPUMM", "SS-Z7PABGL6", "SS-XK8BB3N5",
        "SS-ZBMCG5ZE", "SS-CH8LQL6Y", "SS-FSKU3LZS", "SS-4KTKJW4U", "SS-8Q5N399C",
        "SS-B28XCGYZ", "SS-LSFG8K8D", "SS-HRJLCHA5", "SS-63WPT7Y9", "SS-66S47WAZ",
        "SS-V4P9FVUK", "SS-ER3V8YRJ", "SS-TRUHREYD", "SS-9HSN5FXL", "SS-S3S2UQJU",
        "SS-ZST33ZAD", "SS-TJWC2T23", "SS-C95AZSP9", "SS-LSWAUHCU", "SS-GLTPQP2T",
        "SS-X4V3GVAW", "SS-RZKB2Z8Y", "SS-U94X2LSW", "SS-84KUBH4S", "SS-3TDYK42D",
        "SS-XK6SKWF2", "SS-NZSXSNMU", "SS-6BGFJB56", "SS-CKR9XBFP", "SS-U84E23US",
        "SS-5BFGFHML", "SS-N5MVNDGA", "SS-Q7J8SL9E", "SS-TYJN5RZ8", "SS-QL2W46LP",
        "SS-9CTQF4MD", "SS-DR7XJXMY", "SS-ALEFEPX4", "SS-YVUTR8EC", "SS-ZMZWEHDP",
        "SS-WDD78UTL", "SS-BCZ8X74Z", "SS-R4WHGDC8", "SS-N5MZTP3M", "SS-LLCLNZAQ",
        "SS-JC2HM2QC", "SS-EBGP7LKR", "SS-85RN4E7V", "SS-ZG5BQHF7", "SS-AH5H2B2P",
        "SS-HQWUNT5A", "SS-YNRPF6X2", "SS-7JRDBTTS", "SS-TMMBFRAE", "SS-3J4B8YBD",
        "SS-9N6PQ2AX", "SS-76CDB548", "SS-CGWPDEF6", "SS-KQTCPBME", "SS-E8QQFQAD",
        "SS-4XJ43XRV", "SS-6ETTF2FT", "SS-DZLB74LR", "SS-YWDRRUJR", "SS-KW4KGJME",
        "SS-L6QSFF5Q", "SS-G3JU24NQ", "SS-NZV7SSJG", "SS-9ZGCNG2K", "SS-LWTTVN67",
        "SS-XGABE6KR", "SS-VZDWVTF6", "SS-UPJYWCXR", "SS-GWZZF7WQ", "SS-7KEVLSXA",
        "SS-XZVPP9WX", "SS-NZWV3TT2", "SS-W6YMGGFY", "SS-H7HHUG6G", "SS-VSQUGSPS",
        "SS-WZR4KQQB", "SS-X92R8577", "SS-7UP6SMTR", "SS-WTSBKRF2", "SS-ULNCMKNN",
        "SS-4LXER54F", "SS-UNAC2B3F", "SS-SGACQXNY", "SS-D29SSQVC", "SS-N8PTUDUV",
        "SS-P477V5BL", "SS-46AVWY7E", "SS-QJ7GAMGW", "SS-GJXN4AXU", "SS-SAQTKL7X",
        "SS-THX5MCH8", "SS-RLWK2BD5", "SS-S8WTCT8Q", "SS-2UCX2WUV", "SS-C2LS8SSG",
        "SS-GX4QZA5P", "SS-87JWN4UJ", "SS-4JEUQTNU", "SS-PY85NNE2", "SS-VXV7W4KK",
        "SS-MCUDYFSM", "SS-T22XBFK6", "SS-PLW94RSM", "SS-ZB2JHJG7", "SS-5XH73KTM",
        "SS-92R6G5EN", "SS-8HP896TX", "SS-KPWQRLMA", "SS-R54RSDDM", "SS-WSZYMX2J",
        "SS-4UEGWM7E", "SS-4SX6768L", "SS-EC3AMCYG", "SS-LRQAR2PA", "SS-HRZDMJ67",
        "SS-5E6AGEJU", "SS-YHJTEF29", "SS-VKMRC879", "SS-T8FYYAPK", "SS-SDVX78JB",
        "SS-XVA6TQKP", "SS-JPSTVZDK", "SS-MKB32VA7", "SS-8XRTFM98", "SS-QH2GYLX4",
        "SS-94JTG9JD", "SS-2Q4AQ3H2", "SS-HVYEV9PW", "SS-48MHN2VN", "SS-WDTRU84U",
        "SS-ZFEF4QAK", "SS-ZX4NUVRS", "SS-3RN82AVW", "SS-EPDMMR9Z", "SS-5VDUVNY3"
    ]
    
    for code in season_codes:
        codes[code] = {"total_uses": 200, "used": 0, "type": "season"}
    
    return codes


# Test codes for sandbox work — high use counts so testing never burns
# sellable inventory. Safe to hand out to nobody. Remove any time.
TEST_CODES = {
    "TEST-SANDBOX1": {"total_uses": 100000, "type": "season"},
    "TEST-SANDBOX2": {"total_uses": 100000, "type": "season"},
}

# Initialize codes at startup
ACCESS_CODES = initialize_codes()
for _tc, _meta in TEST_CODES.items():
    ACCESS_CODES[_tc] = {"total_uses": _meta["total_uses"], "used": 0, "type": _meta["type"]}


# --- Request/Response Models ---

class CodeValidationRequest(BaseModel):
    code: str

class ValidateRequest(BaseModel):
    code: str
    device_id: Optional[str] = None

class ConsumeRequest(BaseModel):
    code: Optional[str] = ""          # may be empty during a free period
    item_id: str                      # unique per tagged item; makes edits free
    device_id: Optional[str] = None
    mode: Optional[str] = None        # "manual" | "dictation" | "ai"; older app
                                      # versions omit it, which is treated as "ai"

class AnalyzeRequest(BaseModel):
    code: str
    image1: str  # base64 JPEG — item photo
    image2: Optional[str] = None  # base64 JPEG — label photo
    device_id: Optional[str] = None
    item_id: Optional[str] = None


def code_status(code: str):
    """Look up a code's definition and current remaining uses."""
    entry = ACCESS_CODES.get(code)
    if not entry:
        return None
    total = entry.get("total_uses", 10)
    used = get_used(code)
    return {
        "total": total,
        "used": used,
        "remaining": max(0, total - used),
        "type": entry.get("type", "trial"),
    }


# =============================================================================
# Square -> automatic code delivery
# =============================================================================
# When someone buys EasyTag tags through Square, Square calls /square-webhook.
# We confirm the call really came from Square, look up what was bought, hand out
# unsold codes from the inventory, and email them to the buyer through Resend.
# (Railway's Hobby plan blocks ordinary SMTP email, so an HTTPS email service is
# required.)
#
# Railway variables:
#   SQUARE_ACCESS_TOKEN           Square Developer Console -> your app -> Credentials
#   SQUARE_WEBHOOK_SIGNATURE_KEY  Square Developer Console -> Webhooks -> your subscription
#   SQUARE_WEBHOOK_URL            the EXACT notification URL entered in Square, e.g.
#                                 https://<your-railway-app>.up.railway.app/square-webhook
#   SQUARE_ENVIRONMENT            "production" (default) or "sandbox"
#   SQUARE_CODE_ITEMS             which Square item(s) count as buying an EasyTag
#                                 code (200 tags), e.g.  EasyTag Code
#   SQUARE_TRIAL_ITEMS            optional: Square item(s) that sell a Trial code.
#                                 Leave unset if trials are only given by hand.
#                                 Both are comma-separated; each entry may be the item
#                                 name, "Item: Variation", or a variation ID.
#                                 Not case-sensitive.
#   RESEND_API_KEY                resend.com -> API Keys
#   EMAIL_FROM                    e.g.  Half-Pint Resale <easytag@yourdomain.com>
#                                 (the domain must be verified in Resend)
#   OWNER_EMAIL                   optional: gets a copy of every code email, receives
#                                 buyer replies, and gets an alert if a sale can't
#                                 be fulfilled automatically
#   ADMIN_KEY                     optional: password for the /admin pages below
#   PRESOLD_CODES                 optional: codes already handed out by hand, comma-
#                                 separated, so they are never sold a second time
#   APP_URL                       optional, defaults to the live Netlify site

def _env_set(name: str) -> set:
    return {p.strip().lower() for p in os.environ.get(name, "").split(",") if p.strip()}

SQUARE_ACCESS_TOKEN = os.environ.get("SQUARE_ACCESS_TOKEN", "").strip()
SQUARE_WEBHOOK_SIGNATURE_KEY = os.environ.get("SQUARE_WEBHOOK_SIGNATURE_KEY", "").strip()
SQUARE_WEBHOOK_URL = os.environ.get("SQUARE_WEBHOOK_URL", "").strip()
SQUARE_ENVIRONMENT = os.environ.get("SQUARE_ENVIRONMENT", "production").strip().lower()
SQUARE_API_BASE = (
    "https://connect.squareupsandbox.com" if SQUARE_ENVIRONMENT == "sandbox"
    else "https://connect.squareup.com"
)
# Optional. When unset, Square uses the API version selected for your app.
SQUARE_API_VERSION = os.environ.get("SQUARE_API_VERSION", "").strip()
SQUARE_TRIAL_ITEMS = _env_set("SQUARE_TRIAL_ITEMS")
# The paid 200-tag codes are the "season" type internally (SS- prefix).
# SQUARE_SEASON_ITEMS is the older name for SQUARE_CODE_ITEMS and still works.
SQUARE_SEASON_ITEMS = _env_set("SQUARE_CODE_ITEMS") | _env_set("SQUARE_SEASON_ITEMS")

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()
EMAIL_FROM = os.environ.get("EMAIL_FROM", "").strip()
OWNER_EMAIL = os.environ.get("OWNER_EMAIL", "").strip()
ADMIN_KEY = os.environ.get("ADMIN_KEY", "").strip()
APP_URL = os.environ.get("APP_URL", "https://half-pint-easytag.netlify.app").strip()
PRESOLD_CODES = [
    c.strip().upper()
    for c in os.environ.get("PRESOLD_CODES", "").replace("\n", ",").replace(" ", ",").split(",")
    if c.strip()
]

# What buyers and the admin pages see. Internally the paid codes are type "season".
CODE_LABELS = {"trial": "Trial code", "season": "EasyTag code"}


# Every change to code_inventory takes this one database lock first, so changes
# happen one at a time and can never deadlock each other (e.g. a sale arriving
# while a redeploying server runs its startup cleanup). Sales are rare enough
# that waiting a moment costs nothing.
INVENTORY_LOCK_ID = 7246010


def lock_inventory(cur):
    """Hold the inventory lock until the current transaction commits."""
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (INVENTORY_LOCK_ID,))


def init_code_sales_tables():
    """
    Create the inventory/sales tables and load every sellable code into the
    inventory. Safe to run on every startup: existing rows are never changed,
    except that codes known to be in someone's hands are marked sold.
    """
    if not db_pool:
        return
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            # Take the inventory lock before touching the tables at all, so a
            # second server starting during a redeploy simply waits its turn.
            lock_inventory(cur)

            # One row per sellable code. status: available | sold
            # source says how it left inventory: square | manual | in-use
            cur.execute("""
                CREATE TABLE IF NOT EXISTS code_inventory (
                    code TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'available',
                    source TEXT,
                    payment_id TEXT,
                    buyer_email TEXT,
                    assigned_at TIMESTAMPTZ
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS code_inventory_payment_idx ON code_inventory (payment_id)")
            # Who a hand-given code went to (from the /admin/give-code page).
            cur.execute("ALTER TABLE code_inventory ADD COLUMN IF NOT EXISTS note TEXT")
            # One row per Square payment that bought EasyTag codes.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS square_orders (
                    payment_id TEXT PRIMARY KEY,
                    order_id TEXT,
                    buyer_email TEXT,
                    email_sent_at TIMESTAMPTZ,
                    last_error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            # Load the code list. seq keeps the list order so sales hand out codes
            # from the END of each list, away from any handed out by hand from the top.
            rows = [
                (code, meta["type"], i)
                for i, (code, meta) in enumerate(ACCESS_CODES.items())
                if code not in TEST_CODES
            ]
            cur.executemany(
                "INSERT INTO code_inventory (code, type, seq) VALUES (%s, %s, %s) "
                "ON CONFLICT (code) DO NOTHING",
                rows,
            )

            # Any code that has ever been used in the app is clearly in someone's hands.
            cur.execute("""
                UPDATE code_inventory
                   SET status = 'sold', source = 'in-use', assigned_at = NOW()
                 WHERE status = 'available'
                   AND (code IN (SELECT code FROM code_usage)
                        OR code IN (SELECT code FROM consumed_items))
            """)

            # Codes handed out by hand, listed in the PRESOLD_CODES variable.
            if PRESOLD_CODES:
                cur.execute(
                    """
                    UPDATE code_inventory
                       SET status = 'sold', source = 'manual', assigned_at = NOW()
                     WHERE status = 'available' AND code = ANY(%s)
                    """,
                    (PRESOLD_CODES,),
                )
        conn.commit()
    print("✓ Code inventory ready")


def mark_code_in_use(code: str):
    """A code just showed up in the app, so it must never be sold to anyone else."""
    if not db_pool:
        return
    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                lock_inventory(cur)
                cur.execute(
                    """
                    UPDATE code_inventory
                       SET status = 'sold', source = 'in-use', assigned_at = NOW()
                     WHERE code = %s AND status = 'available'
                    """,
                    (code,),
                )
            conn.commit()
    except Exception as e:
        print(f"!! mark_code_in_use error: {e}")


def verify_square_signature(raw_body: bytes, signature: str) -> bool:
    """
    Square signs each webhook with HMAC-SHA256 over (notification URL + raw body),
    base64-encoded, in the x-square-hmacsha256-signature header.
    """
    if not (SQUARE_WEBHOOK_SIGNATURE_KEY and SQUARE_WEBHOOK_URL and signature):
        return False
    digest = hmac.new(
        SQUARE_WEBHOOK_SIGNATURE_KEY.encode("utf-8"),
        SQUARE_WEBHOOK_URL.encode("utf-8") + raw_body,
        hashlib.sha256,
    ).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


async def square_get(path: str):
    """GET from the Square API. Returns parsed JSON, or None on 404."""
    headers = {
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Accept": "application/json",
    }
    if SQUARE_API_VERSION:
        headers["Square-Version"] = SQUARE_API_VERSION
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(SQUARE_API_BASE + path, headers=headers)
    if r.status_code == 404:
        return None
    if r.status_code >= 300:
        raise RuntimeError(f"Square API {r.status_code} on {path}: {r.text[:300]}")
    return r.json()


def count_codes_in_order(order: dict) -> dict:
    """How many Trial and EasyTag (season-type) codes this Square order paid for."""
    counts = {"trial": 0, "season": 0}
    for li in order.get("line_items") or []:
        name = (li.get("name") or "").strip().lower()
        variation = (li.get("variation_name") or "").strip().lower()
        keys = {
            (li.get("catalog_object_id") or "").strip().lower(),
            name,
            f"{name}: {variation}" if variation else "",
        }
        keys.discard("")
        try:
            qty = max(0, int(float(li.get("quantity") or "1")))
        except (TypeError, ValueError):
            qty = 1
        if keys & SQUARE_TRIAL_ITEMS:
            counts["trial"] += qty
        elif keys & SQUARE_SEASON_ITEMS:
            counts["season"] += qty
    return counts


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


async def find_buyer_email(payment: dict, order: dict) -> Optional[str]:
    """
    Square doesn't always put the buyer's email on the payment, so check, in order:
    the payment, the order's fulfillment recipient, then the customer record.
    """
    email = (payment.get("buyer_email_address") or "").strip()
    if email:
        return email

    for f in order.get("fulfillments") or []:
        for key in ("shipment_details", "pickup_details", "delivery_details", "digital_details"):
            recipient = (f.get(key) or {}).get("recipient") or {}
            email = (recipient.get("email_address") or "").strip()
            if email:
                return email

    # The Half-Pint payment link asks buyers to type their email into a box, which
    # Square stores as a note on the line item ("Email Address: someone@x.com").
    notes = [li.get("note") or "" for li in order.get("line_items") or []]
    notes.append(order.get("note") or "")
    for note in notes:
        m = EMAIL_RE.search(note)
        if m:
            return m.group(0).rstrip(".")

    customer_id = payment.get("customer_id") or order.get("customer_id")
    if customer_id:
        try:
            data = await square_get(f"/v2/customers/{customer_id}")
            email = (((data or {}).get("customer") or {}).get("email_address") or "").strip()
            if email:
                return email
        except Exception as e:
            print(f"!! Square customer lookup failed: {e}")
    return None


class OutOfCodes(Exception):
    pass


def allocate_codes(payment_id: str, buyer_email: Optional[str], counts: dict) -> list:
    """
    Assign unsold codes to this payment, all or nothing. If this payment already
    has codes (a repeated webhook), return those instead of assigning new ones.
    Returns a list of (code, type).
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            # Two webhooks for the same payment arriving at once wait for each other here.
            lock_inventory(cur)
            cur.execute(
                "SELECT code, type FROM code_inventory WHERE payment_id = %s ORDER BY type DESC, code",
                (payment_id,),
            )
            existing = cur.fetchall()
            if existing:
                conn.commit()
                return [(c, t) for c, t in existing]

            assigned = []
            for code_type in ("trial", "season"):
                qty = counts.get(code_type, 0)
                if qty <= 0:
                    continue
                cur.execute(
                    """
                    UPDATE code_inventory
                       SET status = 'sold', source = 'square', payment_id = %s,
                           buyer_email = %s, assigned_at = NOW()
                     WHERE code IN (
                           SELECT code FROM code_inventory
                            WHERE type = %s AND status = 'available'
                            ORDER BY seq DESC
                            LIMIT %s
                              FOR UPDATE SKIP LOCKED)
                    RETURNING code, type
                    """,
                    (payment_id, buyer_email, code_type, qty),
                )
                rows = cur.fetchall()
                if len(rows) < qty:
                    conn.rollback()
                    raise OutOfCodes(f"needed {qty} {CODE_LABELS[code_type]}(s), only {len(rows)} left")
                assigned.extend(rows)
        conn.commit()
    return assigned


def get_square_order(payment_id: str) -> Optional[dict]:
    if not db_pool:
        return None
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT buyer_email, email_sent_at, last_error FROM square_orders WHERE payment_id = %s",
                (payment_id,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {"buyer_email": row[0], "email_sent_at": row[1], "last_error": row[2]}


def record_square_order(payment_id: str, order_id: Optional[str], buyer_email: Optional[str],
                        error: Optional[str] = None, email_sent: bool = False):
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO square_orders (payment_id, order_id, buyer_email, last_error, email_sent_at)
                VALUES (%s, %s, %s, %s, CASE WHEN %s THEN NOW() END)
                ON CONFLICT (payment_id) DO UPDATE SET
                    order_id = COALESCE(EXCLUDED.order_id, square_orders.order_id),
                    buyer_email = COALESCE(EXCLUDED.buyer_email, square_orders.buyer_email),
                    last_error = EXCLUDED.last_error,
                    email_sent_at = CASE WHEN %s THEN NOW() ELSE square_orders.email_sent_at END,
                    updated_at = NOW()
                """,
                (payment_id, order_id, buyer_email, error, email_sent, email_sent),
            )
            if buyer_email:
                lock_inventory(cur)
                cur.execute(
                    "UPDATE code_inventory SET buyer_email = %s WHERE payment_id = %s",
                    (buyer_email, payment_id),
                )
        conn.commit()


async def send_email(to: str, subject: str, html_body: str, text_body: str,
                     idempotency_key: Optional[str] = None, copy_owner: bool = True):
    """Send one email through Resend's HTTPS API. Raises on failure."""
    if not (RESEND_API_KEY and EMAIL_FROM):
        raise RuntimeError("RESEND_API_KEY / EMAIL_FROM not set")
    payload = {"from": EMAIL_FROM, "to": [to], "subject": subject,
               "html": html_body, "text": text_body}
    if OWNER_EMAIL:
        payload["reply_to"] = OWNER_EMAIL
        if copy_owner and OWNER_EMAIL.lower() != to.lower():
            payload["bcc"] = [OWNER_EMAIL]
    headers = {"Authorization": f"Bearer {RESEND_API_KEY}"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post("https://api.resend.com/emails", json=payload, headers=headers)
    if r.status_code >= 300:
        raise RuntimeError(f"Resend {r.status_code}: {r.text[:300]}")


async def alert_owner(subject: str, message: str):
    """Best-effort heads-up to OWNER_EMAIL. Never raises."""
    print(f"!! {subject}: {message}")
    if not OWNER_EMAIL:
        return
    try:
        await send_email(OWNER_EMAIL, f"[EasyTag] {subject}",
                         f"<pre>{html_lib.escape(message)}</pre>", message, copy_owner=False)
    except Exception as e:
        print(f"!! Owner alert failed: {e}")


def build_code_email(codes: list) -> tuple:
    """Returns (subject, html, text) for the buyer's code email."""
    plural = len(codes) > 1
    subject = "Your Half-Pint EasyTag codes" if plural else "Your Half-Pint EasyTag code"

    lines_text, lines_html = [], []
    for code, code_type in codes:
        uses = ACCESS_CODES.get(code, {}).get("total_uses", 0)
        label = CODE_LABELS.get(code_type, "Code")
        lines_text.append(f"  {code}   ({label}, {uses} tags)")
        lines_html.append(
            f'<tr><td style="font:bold 22px monospace;letter-spacing:1px;padding:6px 16px 6px 0">'
            f'{html_lib.escape(code)}</td>'
            f'<td style="color:#555">{label}, {uses} tags</td></tr>'
        )

    code_word = "codes" if plural else "code"
    text = (
        f"Thanks for buying EasyTag tags for Half-Pint Resale!\n\n"
        f"Your {code_word}:\n" + "\n".join(lines_text) + "\n\n"
        f"How to use it:\n"
        f"1. Open EasyTag in Safari or Chrome: {APP_URL}\n"
        f"   It won't work inside Facebook, Google or other in-app browsers.\n"
        f"   On a phone, add it to your home screen so your tagged items stay saved.\n"
        f"2. Tap Dictation or AI Auto-Fill. When the app asks for an access code, enter the code above.\n\n"
        f"Each item tagged with Dictation or AI Auto-Fill uses one tag. Manual Entry is always free, "
        f"and editing an item you've already tagged doesn't use another tag.\n\n"
        f"Questions? Just reply to this email.\n\n"
        f"Half-Pint Resale\n"
    )
    url = html_lib.escape(APP_URL)
    html_body = (
        '<div style="font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.5;color:#222;max-width:560px">'
        "<p>Thanks for buying EasyTag tags for Half-Pint Resale!</p>"
        f"<p>Your {code_word}:</p>"
        f'<table style="border-collapse:collapse;margin:0 0 16px">{"".join(lines_html)}</table>'
        "<p><b>How to use it</b></p><ol>"
        f'<li>Open EasyTag in Safari or Chrome: <a href="{url}">{url}</a><br>'
        "It won't work inside Facebook, Google or other in-app browsers. "
        "On a phone, add it to your home screen so your tagged items stay saved.</li>"
        "<li>Tap <b>Dictation</b> or <b>AI Auto-Fill</b>. When the app asks for an access code, enter the code above.</li>"
        "</ol>"
        "<p>Each item tagged with Dictation or AI Auto-Fill uses one tag. Manual Entry is always free, "
        "and editing an item you've already tagged doesn't use another tag.</p>"
        "<p>Questions? Just reply to this email.</p>"
        "<p>Half-Pint Resale</p></div>"
    )
    return subject, html_body, text


# A webhook for a payment older than this is an update to an old sale (a refund,
# a note edit), not a new purchase. Square's own retries finish well inside this.
FRESH_PAYMENT_HOURS = 72


def _payment_age_hours(payment: dict) -> Optional[float]:
    created = (payment.get("created_at") or "").strip()
    if not created:
        return None
    try:
        when = datetime.fromisoformat(created.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - when).total_seconds() / 3600
    except ValueError:
        return None


def _is_refunded(payment: dict) -> bool:
    return int(((payment.get("refunded_money") or {}).get("amount")) or 0) > 0


async def fulfill_payment(payment: dict, email_override: Optional[str] = None,
                          force_resend: bool = False, from_webhook: bool = False) -> dict:
    """
    Turn one completed Square payment into emailed codes. Safe to call any number
    of times for the same payment: codes are assigned once and the email is sent once
    (unless force_resend).
    """
    payment_id = payment.get("id")
    order_id = payment.get("order_id")
    if not payment_id:
        return {"ok": True, "skipped": "no payment id"}
    if payment.get("status") != "COMPLETED":
        return {"ok": True, "skipped": f"payment status {payment.get('status')}"}
    if not order_id:
        return {"ok": True, "skipped": "payment has no order"}
    if _is_refunded(payment):
        return {"ok": True, "skipped": "payment was refunded"}
    if from_webhook:
        age = _payment_age_hours(payment)
        if age is not None and age > FRESH_PAYMENT_HOURS:
            # Old sales (including ones whose codes were sent by hand before this
            # went live) never get an automatic code. /admin/retry-payment still can.
            return {"ok": True, "skipped": "update to an old payment"}

    # Square sends payment.updated more than once per payment. Once the email has
    # gone out, there's nothing more to do.
    prev = get_square_order(payment_id)
    if prev and prev["email_sent_at"] and not force_resend:
        return {"ok": True, "already_sent": True}

    order = ((await square_get(f"/v2/orders/{order_id}")) or {}).get("order")
    if not order:
        return {"ok": True, "skipped": "order not found"}

    counts = count_codes_in_order(order)
    if not (counts["trial"] or counts["season"]):
        return {"ok": True, "skipped": "no EasyTag items in order"}

    email = (email_override or "").strip() or await find_buyer_email(payment, order)

    try:
        codes = allocate_codes(payment_id, email, counts)
    except OutOfCodes as e:
        error = f"Out of codes: {e}"
        if not prev or prev["last_error"] != error:
            await alert_owner(
                "Sale could not be filled: out of codes",
                f"Square payment {payment_id} (order {order_id}) bought "
                f"{counts['season']} EasyTag code(s) and {counts['trial']} trial code(s), "
                f"but there aren't enough unsold codes.\nBuyer email: {email or 'unknown'}\n\n"
                f"Add codes to main.py, redeploy, then open "
                f"/admin/retry-payment?payment_id={payment_id}&key=YOUR_ADMIN_KEY",
            )
        record_square_order(payment_id, order_id, email, error=error)
        return {"ok": False, "error": "out_of_codes", "detail": str(e)}

    code_list = ", ".join(c for c, _ in codes)

    if not email:
        error = "No buyer email found on the Square payment, order or customer"
        if not prev or prev["last_error"] != error:
            await alert_owner(
                "Sale needs a manual email",
                f"Square payment {payment_id} (order {order_id}) was assigned: {code_list}\n"
                f"but Square didn't provide the buyer's email.\n\n"
                f"Either email the code yourself, or open\n"
                f"/admin/retry-payment?payment_id={payment_id}&email=BUYER@EXAMPLE.COM&key=YOUR_ADMIN_KEY",
            )
        record_square_order(payment_id, order_id, None, error=error)
        return {"ok": False, "error": "no_email", "codes": [c for c, _ in codes]}

    subject, html_body, text_body = build_code_email(codes)
    try:
        await send_email(
            email, subject, html_body, text_body,
            # Resend drops a repeat of the same key, so a retried webhook can't
            # double-send. A deliberate resend skips the key.
            idempotency_key=None if force_resend else f"easytag-codes-{payment_id}",
        )
    except Exception as e:
        record_square_order(payment_id, order_id, email, error=f"Email failed: {e}"[:500])
        raise  # webhook answers 500 so Square tries again later

    record_square_order(payment_id, order_id, email, error=None, email_sent=True)
    print(f"✓ Square sale {payment_id}: sent {code_list} to {email}")
    return {"ok": True, "sent_to": email, "codes": [c for c, _ in codes]}


def require_admin(request: Request, form_key: Optional[str] = None):
    key = form_key or request.headers.get("x-admin-key") or request.query_params.get("key") or ""
    if not ADMIN_KEY or not hmac.compare_digest(key, ADMIN_KEY):
        raise HTTPException(status_code=403, detail="Admin key required.")


def take_code_by_hand(code_type: str, note: str, email: Optional[str]) -> Optional[str]:
    """
    Mark the next free code of this type as given out by hand and return it.
    Takes from the TOP of the list; Square sales take from the bottom.
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            lock_inventory(cur)
            cur.execute(
                """
                UPDATE code_inventory
                   SET status = 'sold', source = 'manual', note = %s,
                       buyer_email = %s, assigned_at = NOW()
                 WHERE code = (
                       SELECT code FROM code_inventory
                        WHERE type = %s AND status = 'available'
                        ORDER BY seq ASC
                        LIMIT 1
                          FOR UPDATE SKIP LOCKED)
                RETURNING code
                """,
                (note or None, email or None, code_type),
            )
            row = cur.fetchone()
        conn.commit()
    return row[0] if row else None


def _admin_page(title: str, body: str):
    from fastapi.responses import HTMLResponse
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html_lib.escape(title)}</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:480px;margin:24px auto;padding:0 16px;color:#222}"
        "label{display:block;margin:14px 0 4px;font-weight:bold}"
        "input,select,button{font-size:17px;padding:10px;width:100%;box-sizing:border-box}"
        "button{margin-top:20px;background:#2a6;color:#fff;border:0;border-radius:6px}"
        ".code{font:bold 30px monospace;letter-spacing:2px;padding:14px;background:#f3f3f3;"
        "border-radius:6px;text-align:center;margin:12px 0;user-select:all}"
        ".muted{color:#666;font-size:14px}.err{color:#b00}</style></head><body>"
        f"{body}</body></html>"
    )


# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    init_database()
    try:
        init_code_sales_tables()
    except Exception as e:
        print(f"!! Code inventory setup error: {e}")
    print(f"✓ Loaded {len(ACCESS_CODES)} access codes")
    trial_count = sum(1 for c in ACCESS_CODES.values() if c["type"] == "trial")
    season_count = sum(1 for c in ACCESS_CODES.values() if c["type"] == "season")
    print(f"  - {trial_count} trial codes (10 uses each)")
    print(f"  - {season_count} EasyTag codes (200 uses each)")
    print(f"  - persistence: {'ON (database)' if db_pool else 'OFF (in-memory only!)'}")

@app.get("/")
async def root():
    return {"message": "Half-Pint Resale API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok", "database": bool(db_pool)}

@app.get("/status")
async def status():
    """
    App configuration, checked on load.

    The app uses free_now to decide whether tagging is open to everyone. This is
    evaluated on the server clock so changing the date on a phone has no effect.
    """
    fp = free_period_status()
    return {
        "free_now": fp["free_now"],
        "free_start": fp["start"],
        "free_end": fp["end"],
        "free_includes_ai": FREE_PERIOD_INCLUDES_AI,
        "dictation_free": DICTATION_FREE,
        "database": bool(db_pool),
    }


@app.post("/validate-code")
async def validate_code(request: ValidateRequest):
    """Validate an access code and return remaining uses."""
    code = request.code.strip().upper()

    status = code_status(code)
    if not status:
        raise HTTPException(status_code=404, detail="Invalid access code. Please check and try again.")

    # Someone has this code in hand, so Square sales must never hand it out again.
    mark_code_in_use(code)

    return {
        "valid": True,
        "uses_remaining": status["remaining"],
        "total_uses": status["total"],
        "type": status["type"],
    }


@app.post("/consume")
async def consume(request: ConsumeRequest):
    """
    Deduct one use for a newly created tag.

    Charging is keyed on item_id, so editing an existing item costs nothing and
    a retried request can never double-charge.

    Free entry methods never reach this endpoint from a current app version —
    the checks below are a backstop against a stale cached copy of the app.
    """
    code = (request.code or "").strip().upper()
    item_id = (request.item_id or "").strip()
    mode = (request.mode or "").strip().lower()

    if not item_id:
        raise HTTPException(status_code=400, detail="item_id is required.")

    # --- Always-free entry methods: manual typing, and dictation when it is
    # switched on as free. Recorded at zero cost so a later edit stays free. ---
    if mode == "manual" or (mode == "dictation" and DICTATION_FREE):
        free_code = code if code else "FREE-MODE"
        if db_pool:
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO consumed_items (code, item_id)
                        VALUES (%s, %s)
                        ON CONFLICT (code, item_id) DO NOTHING
                        """,
                        (free_code, item_id),
                    )
                conn.commit()
        status = code_status(code) if code else None
        return {
            "ok": True,
            "charged": False,
            "free_mode": mode,
            "uses_remaining": status["remaining"] if status else None,
            "total_uses": status["total"] if status else None,
        }

    # --- Free period: tagging is open to everyone, nothing is deducted ---
    # The item is still recorded as charged (at zero cost) so that editing it
    # later stays free even after the free window closes.
    fp = free_period_status()
    if fp["free_now"]:
        free_code = code if code else "FREE-PERIOD"
        if db_pool:
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO consumed_items (code, item_id)
                        VALUES (%s, %s)
                        ON CONFLICT (code, item_id) DO NOTHING
                        """,
                        (free_code, item_id),
                    )
                conn.commit()
        status = code_status(code) if code else None
        return {
            "ok": True,
            "charged": False,
            "free_period": True,
            "uses_remaining": status["remaining"] if status else None,
            "total_uses": status["total"] if status else None,
        }

    status = code_status(code)
    if not status:
        raise HTTPException(status_code=404, detail="Invalid access code.")

    # If this item was already charged, report success without deducting again.
    already_charged = False
    if db_pool:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM consumed_items WHERE code = %s AND item_id = %s",
                    (code, item_id),
                )
                already_charged = cur.fetchone() is not None

    if already_charged:
        return {
            "ok": True,
            "charged": False,
            "uses_remaining": status["remaining"],
            "total_uses": status["total"],
        }

    if status["remaining"] <= 0:
        raise HTTPException(status_code=403, detail="No tags remaining on this code.")

    new_used = consume_use(code, item_id)
    remaining = max(0, status["total"] - new_used)

    return {
        "ok": True,
        "charged": True,
        "uses_remaining": remaining,
        "total_uses": status["total"],
    }


@app.post("/analyze-photo")
async def analyze_photo(request: AnalyzeRequest):
    """
    Analyze two photos using AI and return structured clothing item data.
    Photo 1 (image1): The clothing item — analyzed by OpenAI Vision
    Photo 2 (image2): The clothing label — analyzed by Google Cloud Vision OCR
    """
    code = request.code.strip().upper()

    # AI photos are free during the window only if FREE_PERIOD_INCLUDES_AI is on,
    # because each call costs real money at OpenAI and Google.
    fp = free_period_status()
    ai_is_free = fp["free_now"] and FREE_PERIOD_INCLUDES_AI

    # Validate code
    status = code_status(code)
    if not status:
        if ai_is_free:
            # No code needed while AI is free
            status = {"total": 0, "used": 0, "remaining": 0, "type": "free"}
        else:
            raise HTTPException(status_code=403, detail="Invalid or expired access code.")

    total = status["total"]
    remaining = status["remaining"]

    if remaining <= 0 and not ai_is_free:
        raise HTTPException(status_code=403, detail="No tags remaining on this code.")

    # --- Step 1: Analyze item photo with OpenAI Vision ---
    item_data = {"brand": "", "description": "", "size": "", "gender": "neutral"}

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a children's clothing item for a consignment sale. "
                                "Look at the item and respond ONLY with a JSON object (no markdown, no explanation) with these keys:\n"
                                "- brand: brand name if visible on item, otherwise empty string\n"
                                "- description: concise description with primary color/pattern + any visible brand text/logos on the item + item type (e.g. 'grey Nike sweatpants', 'pink floral dress', 'blue striped polo'). Keep it 2-4 words.\n"
                                "- size: size if visible on item tag/label (e.g. '4T', '2T', 'XS', '6-12m'), otherwise empty string\n"
                                "- gender: 'boy', 'girl', or 'neutral' based on the item's style\n"
                                "Examples: {\"brand\": \"Nike\", \"description\": \"grey Nike sweatpants\", \"size\": \"12-13 YRS\", \"gender\": \"neutral\"}, {\"brand\": \"Carter's\", \"description\": \"pink floral dress\", \"size\": \"4T\", \"gender\": \"girl\"}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{request.image1}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        item_data.update({
            "brand": parsed.get("brand", "").title(),
            "description": parsed.get("description", "").title(),
            "size": parsed.get("size", ""),
            "gender": parsed.get("gender", "neutral")
        })
    except Exception as e:
        print(f"OpenAI Vision error: {e}")
        # Non-fatal — continue with label photo

    # --- Step 2: Extract text from label photo with Google Cloud Vision ---
    try:
        vision_client = get_vision_client()
        if vision_client and request.image2:
            from google.cloud import vision as gvision
            image_bytes = base64.b64decode(request.image2)
            image = gvision.Image(content=image_bytes)
            result = vision_client.text_detection(image=image)

            if result.text_annotations:
                label_text = result.text_annotations[0].description

                # Use GPT to parse the label text into structured data
                parse_response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=150,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"This text was extracted from a children's clothing label:\n\n{label_text}\n\n"
                                "Extract and respond ONLY with a JSON object (no markdown) with these keys:\n"
                                "- brand: brand name from label, or empty string\n"
                                "- size: size from label (e.g. '4T', '2T', 'XS', '6-12m'), or empty string\n"
                                "Example: {\"brand\": \"Carter's\", \"size\": \"4T\"}"
                            )
                        }
                    ]
                )

                raw2 = parse_response.choices[0].message.content.strip()
                if raw2.startswith("```"):
                    raw2 = raw2.split("```")[1]
                    if raw2.startswith("json"):
                        raw2 = raw2[4:]
                label_data = json.loads(raw2)

                # Label data overrides item photo data (more reliable)
                if label_data.get("brand"):
                    item_data["brand"] = label_data["brand"].title()
                if label_data.get("size"):
                    item_data["size"] = label_data["size"]

    except Exception as e:
        print(f"Google Vision / label parse error: {e}")
        # Non-fatal — use whatever we got from photo 1

    # --- Increment usage (persisted) ---
    # Photo analysis charges the same single use as any other tag. If the app
    # passed an item_id, charging is idempotent for that item.
    if ai_is_free:
        new_remaining = remaining
    else:
        new_used = consume_use(code, request.item_id)
        new_remaining = max(0, total - new_used)

    return {
        "brand": item_data.get("brand", ""),
        "description": item_data.get("description", ""),
        "size": item_data.get("size", ""),
        "gender": item_data.get("gender", "neutral"),
        "uses_remaining": new_remaining
    }


@app.post("/square-webhook")
async def square_webhook(request: Request):
    """
    Square calls this for every payment event. Only completed payments whose order
    contains an EasyTag item do anything; everything else is acknowledged and ignored.
    """
    raw = await request.body()
    signature = request.headers.get("x-square-hmacsha256-signature", "")
    if not verify_square_signature(raw, signature):
        print("!! Square webhook rejected: signature did not match "
              "(check SQUARE_WEBHOOK_SIGNATURE_KEY and SQUARE_WEBHOOK_URL)")
        raise HTTPException(status_code=403, detail="Invalid signature.")

    try:
        event = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Bad JSON.")

    if event.get("type") != "payment.updated":
        return {"ok": True, "ignored": event.get("type")}

    payment = ((event.get("data") or {}).get("object") or {}).get("payment") or {}

    if not db_pool:
        # Without the database we can't track inventory; let Square retry later.
        raise HTTPException(status_code=503, detail="Database unavailable.")

    try:
        result = await fulfill_payment(payment, from_webhook=True)
    except Exception as e:
        print(f"!! Square fulfillment error for payment {payment.get('id')}: {e}")
        try:
            if payment.get("id"):
                record_square_order(payment["id"], payment.get("order_id"), None,
                                    error=f"{e}"[:500])
        except Exception:
            pass
        # A non-2xx answer makes Square deliver this event again later.
        raise HTTPException(status_code=500, detail="Fulfillment failed; will retry.")

    if not result.get("skipped"):
        print(f"Square webhook {payment.get('id')}: {result}")
    return result


@app.get("/admin/code-sales")
async def admin_code_sales(request: Request):
    """
    Inventory counts, setup check, and the 50 most recent Square sales.
    Open in a browser:  /admin/code-sales?key=YOUR_ADMIN_KEY
    """
    require_admin(request)
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT type, status, COALESCE(source, ''), COUNT(*)
                  FROM code_inventory GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
            """)
            inventory = {}
            for code_type, status, source, n in cur.fetchall():
                t = inventory.setdefault(code_type, {"available": 0, "sold": {}})
                if status == "available":
                    t["available"] += n
                else:
                    t["sold"][source or "other"] = n

            cur.execute("""
                SELECT o.payment_id, o.order_id, o.buyer_email, o.email_sent_at,
                       o.last_error, o.created_at,
                       COALESCE(string_agg(i.code, ', ' ORDER BY i.code), '')
                  FROM square_orders o
                  LEFT JOIN code_inventory i ON i.payment_id = o.payment_id
                 GROUP BY o.payment_id
                 ORDER BY o.created_at DESC
                 LIMIT 50
            """)
            sales = [
                {
                    "payment_id": r[0], "order_id": r[1], "buyer_email": r[2],
                    "email_sent_at": r[3], "problem": r[4], "created_at": r[5],
                    "codes": r[6],
                }
                for r in cur.fetchall()
            ]

            cur.execute("""
                SELECT code, type, note, buyer_email, assigned_at
                  FROM code_inventory
                 WHERE source = 'manual' AND note IS NOT NULL
                 ORDER BY assigned_at DESC
                 LIMIT 50
            """)
            hand_given = [
                {"code": r[0], "type": r[1], "note": r[2], "email": r[3], "given_at": r[4]}
                for r in cur.fetchall()
            ]

    return {
        "setup": {
            "square_access_token": bool(SQUARE_ACCESS_TOKEN),
            "square_signature_key": bool(SQUARE_WEBHOOK_SIGNATURE_KEY),
            "square_webhook_url": SQUARE_WEBHOOK_URL or None,
            "square_environment": SQUARE_ENVIRONMENT,
            "trial_items": sorted(SQUARE_TRIAL_ITEMS),
            "easytag_code_items": sorted(SQUARE_SEASON_ITEMS),
            "resend_api_key": bool(RESEND_API_KEY),
            "email_from": EMAIL_FROM or None,
            "owner_email": OWNER_EMAIL or None,
            "presold_codes_listed": len(PRESOLD_CODES),
        },
        "inventory": inventory,
        "recent_sales": sales,
        "recent_hand_given": hand_given,
    }


@app.get("/admin/give-code")
async def admin_give_code_form(request: Request):
    """
    Page for handing out a code by hand (thank-yous, direct payments).
    Bookmark:  /admin/give-code?key=YOUR_ADMIN_KEY
    Opening the page never uses up a code; only the button does.
    """
    require_admin(request)
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT type, COUNT(*) FROM code_inventory WHERE status = 'available' GROUP BY type")
            left = dict(cur.fetchall())
    key = html_lib.escape(request.query_params.get("key", ""))
    return _admin_page("Give out a code", f"""
        <h2>Give out an EasyTag code</h2>
        <form method="post" action="/admin/give-code" onsubmit="this.querySelector('button').disabled=true">
          <input type="hidden" name="key" value="{key}">
          <label>Type</label>
          <select name="type">
            <option value="season">EasyTag code (200 tags), {left.get('season', 0)} left</option>
            <option value="trial">Trial code (10 tags), {left.get('trial', 0)} left</option>
          </select>
          <label>Who is it for? (note)</label>
          <input name="note" placeholder="e.g. Brittany, thank-you" required>
          <label>Their email (optional)</label>
          <input name="email" type="email" placeholder="Leave blank to just show the code">
          <p class="muted">If you fill in an email, they get the same code email a Square buyer gets.</p>
          <button type="submit">Get code</button>
        </form>""")


@app.post("/admin/give-code")
async def admin_give_code(request: Request):
    from urllib.parse import parse_qs
    form = {k: v[0].strip() for k, v in parse_qs((await request.body()).decode("utf-8")).items()}
    require_admin(request, form_key=form.get("key"))
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    code_type = form.get("type") if form.get("type") in ("trial", "season") else "season"
    note = form.get("note", "")[:200]
    email = form.get("email", "")[:200]
    key = html_lib.escape(form.get("key", ""))
    again = f'<p><a href="/admin/give-code?key={key}">Give out another code</a></p>'

    code = take_code_by_hand(code_type, note, email)
    if not code:
        return _admin_page("Out of codes",
            f"<h2 class='err'>No {CODE_LABELS[code_type]}s left</h2>"
            "<p>Add more codes to main.py and redeploy.</p>" + again)

    sent = ""
    if email:
        subject, html_body, text_body = build_code_email([(code, code_type)])
        try:
            await send_email(email, subject, html_body, text_body,
                             idempotency_key=f"easytag-manual-{code}")
            sent = f"<p>✓ Emailed to <b>{html_lib.escape(email)}</b>.</p>"
        except Exception as e:
            print(f"!! Manual code email failed: {e}")
            sent = (f"<p class='err'>The email to {html_lib.escape(email)} didn't go through. "
                    "The code is still reserved for them; send it yourself.</p>")

    uses = ACCESS_CODES.get(code, {}).get("total_uses", 0)
    return _admin_page("Code ready", f"""
        <h2>{CODE_LABELS[code_type]} ({uses} tags)</h2>
        <div class="code">{html_lib.escape(code)}</div>
        <p class="muted">For: {html_lib.escape(note) or '(no note)'}. This code is now marked as given out
        and will never be sold through Square.</p>
        {sent}{again}""")


@app.get("/admin/retry-payment")
async def admin_retry_payment(request: Request, payment_id: str,
                              email: Optional[str] = None, resend: bool = False):
    """
    Re-run delivery for one Square payment, e.g. after an "out of codes" or
    "no email" alert. Never assigns a second set of codes to the same payment.
      /admin/retry-payment?payment_id=XXXX&key=YOUR_ADMIN_KEY
      add &email=buyer@example.com to send to a specific address
      add &resend=true to send the same codes again
    """
    require_admin(request)
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    data = await square_get(f"/v2/payments/{payment_id.strip()}")
    payment = (data or {}).get("payment")
    if not payment:
        raise HTTPException(status_code=404, detail="Square payment not found.")
    try:
        return await fulfill_payment(payment, email_override=email,
                                     force_resend=resend or bool(email))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Delivery failed: {e}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
