from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import base64
import json
from typing import Optional

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

# Initialize codes at startup
ACCESS_CODES = initialize_codes()

# --- Request/Response Models ---

class CodeValidationRequest(BaseModel):
    code: str

class ValidateRequest(BaseModel):
    code: str
    device_id: Optional[str] = None

class AnalyzeRequest(BaseModel):
    code: str
    image1: str  # base64 JPEG — item photo
    image2: Optional[str] = None  # base64 JPEG — label photo
    device_id: Optional[str] = None

# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    print(f"✓ Loaded {len(ACCESS_CODES)} access codes")
    trial_count = sum(1 for c in ACCESS_CODES.values() if c["type"] == "trial")
    season_count = sum(1 for c in ACCESS_CODES.values() if c["type"] == "season")
    print(f"  - {trial_count} trial codes (10 uses each)")
    print(f"  - {season_count} season codes (200 uses each)")

@app.get("/")
async def root():
    return {"message": "Half-Pint Resale API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/validate-code")
async def validate_code(request: ValidateRequest):
    """Validate an access code and return remaining uses."""
    code = request.code.strip().upper()

    if code not in ACCESS_CODES:
        raise HTTPException(status_code=404, detail="Invalid access code. Please check and try again.")

    entry = ACCESS_CODES[code]
    total = entry.get("total_uses", 10)
    used = entry.get("used", 0)
    remaining = max(0, total - used)

    return {
        "valid": True,
        "uses_remaining": remaining,
        "total_uses": total,
        "type": entry.get("type", "trial")
    }

@app.post("/analyze-photo")
async def analyze_photo(request: AnalyzeRequest):
    """
    Analyze two photos using AI and return structured clothing item data.
    Photo 1 (image1): The clothing item — analyzed by OpenAI Vision
    Photo 2 (image2): The clothing label — analyzed by Google Cloud Vision OCR
    """
    code = request.code.strip().upper()

    # Validate code
    if code not in ACCESS_CODES:
        raise HTTPException(status_code=403, detail="Invalid or expired access code.")

    entry = ACCESS_CODES[code]
    total = entry.get("total_uses", 10)
    used = entry.get("used", 0)
    remaining = max(0, total - used)

    if remaining <= 0:
        raise HTTPException(status_code=403, detail="No uses remaining on this code.")

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

    # --- Increment usage ---
    ACCESS_CODES[code]["used"] = used + 1
    new_remaining = max(0, total - (used + 1))

    return {
        "brand": item_data.get("brand", ""),
        "description": item_data.get("description", ""),
        "size": item_data.get("size", ""),
        "gender": item_data.get("gender", "neutral"),
        "uses_remaining": new_remaining
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
