"""Dati sì/no sui controlli delle azioni di un agente (fase 2), da tldr-pages (CC BY 4.0, revisione fissata).

Ogni esempio di tldr-pages è un comando reale con una descrizione (in inglese e, quando esiste la traduzione, in
it/es/de/fr). Le 6 proprietà del test v0 (bench/build_ood_v0.py: PROPS) sono etichettate con REGOLE sul comando
(funzione `labels`): 1 / 0 quando la regola è sicura, assente altrimenti. Nessun testo generato.

Divisione per programma (pagina tldr), con hash stabile: ~10% test, ~10% sviluppo, il resto addestramento; così test e
addestramento non condividono mai lo stesso programma. Testi: solo comando, solo descrizione (nella lingua della
traduzione), oppure "descrizione: comando". La proprietà "usare privilegi di amministratore" si etichetta solo se il
comando è nel testo.

uso: python bench/build_agent_actions.py --tldr data/tldr [--download] --out bench/data/agent
"""
import argparse, collections, hashlib, json, os, random, re, shlex

TLDR = ("tldr-pages/tldr", "b9650ebb236ac9ca2ce11a5f1f15d72794f36c4a")
PLATFORMS = ("common", "linux", "osx")
LANGS = ("it", "es", "de", "fr")
PROPS = ("del", "net", "sudo", "inst", "ro", "push")

READONLY = set("""ls cat less more head tail grep egrep fgrep rg ag ack find locate which whereis type file stat wc du df free
uptime uname whoami id groups ps top htop btop pstree lsof lsblk lscpu lsusb lspci lsmod env printenv date cal pwd tree diff
cmp md5sum sha1sum sha256sum sha512sum b2sum cksum uniq cut jq yq history last lastlog w who dmesg journalctl netstat ss
ping ping6 dig nslookup host whois traceroute tracepath mtr man tldr hexdump xxd od strings nm ldd readelf objdump
column fold nl tac rev comm join paste expr bc factor seq hostname nproc getconf vmstat iostat mpstat sar sensors
arch printf tput exa eza bat fd zcat zless zgrep bzcat xzcat""".split())
LOCAL_MODIFY = set("""mkdir touch cp mv ln chmod chown chgrp rm rmdir shred unlink tar zip unzip gzip gunzip bzip2 xz unxz 7z
tee kill killall pkill reboot shutdown poweroff halt service mount umount crontab useradd userdel usermod passwd groupadd
truncate dd mkfs wipefs fdisk parted sfdisk rename install ffmpeg convert magick mogrify patch split csplit rsync
chattr setfacl sysctl modprobe rmmod insmod hwclock timedatectl hostnamectl localectl swapon swapoff""".split())
NET_PROGS = set("""curl wget ssh scp sftp ping ping6 dig nslookup host whois traceroute tracepath mtr ftp telnet nc ncat
netcat http https xh aria2c yt-dlp youtube-dl rclone aws gcloud az gh mosh sshpass""".split())
PKG = {"apt": {"install", "upgrade", "full-upgrade", "dist-upgrade", "reinstall"},
       "apt-get": {"install", "upgrade", "dist-upgrade", "reinstall"}, "aptitude": {"install", "upgrade"},
       "dnf": {"install", "upgrade", "update", "reinstall"}, "yum": {"install", "update", "upgrade", "reinstall"},
       "zypper": {"install", "in", "update", "up"}, "brew": {"install", "upgrade", "reinstall"},
       "pip": {"install"}, "pip3": {"install"}, "pipx": {"install"}, "npm": {"install", "i", "update", "ci"},
       "yarn": {"add", "install", "upgrade"}, "pnpm": {"add", "install", "update", "i"}, "cargo": {"install"},
       "gem": {"install", "update"}, "go": {"install", "get"}, "snap": {"install", "refresh"},
       "flatpak": {"install", "update"}, "conda": {"install", "create", "update"}, "mamba": {"install", "create"},
       "port": {"install", "upgrade"}, "winget": {"install", "upgrade"}, "choco": {"install", "upgrade"}}
PKG_RO = {"list", "search", "show", "info", "view", "ls", "outdated", "freeze", "policy", "query"}
PUSH = {("git", "push"), ("npm", "publish"), ("yarn", "publish"), ("pnpm", "publish"), ("cargo", "publish"),
        ("gem", "push"), ("docker", "push"), ("podman", "push"), ("twine", "upload"), ("poetry", "publish"),
        ("helm", "push")}
GIT_RO = {"log", "status", "diff", "show", "blame", "shortlog", "describe", "rev-parse", "grep", "ls-files", "reflog"}
GIT_NET = {"clone", "push", "pull", "fetch", "ls-remote", "submodule"}
GIT_MOD = {"add", "commit", "merge", "rebase", "checkout", "switch", "reset", "restore", "stash", "tag", "cherry-pick",
           "revert", "rm", "mv", "clean", "init", "am", "apply", "branch", "config", "push", "pull", "clone", "fetch"}
DOCKER_RO = {"ps", "images", "logs", "inspect", "version", "info", "stats", "top", "history"}
REMOTE = set("ssh sftp mosh sshpass telnet nc ncat netcat ftp aws gcloud az gh rclone".split())
URL = re.compile(r"https?://|ftp://|\b[\w.-]+@[\w.-]+:")


def render(cmd, rng):
    """{{path/to/file}} -> path/to/file; {{[-a|--all]}} -> una delle alternative."""
    def rep(m):
        x = m.group(1)
        alt = re.fullmatch(r"\[(.+)\]", x)
        if alt:
            return rng.choice(alt.group(1).split("|"))
        return x
    return re.sub(r"\{\{(.*?)\}\}", rep, cmd)


def segments(cmd):
    out = []
    for seg in re.split(r"\|\||&&|\||;", cmd):
        try:
            toks = shlex.split(seg, posix=True)
        except ValueError:
            toks = seg.split()
        if not toks:
            continue
        sudo = False
        while toks and toks[0] in ("sudo", "doas", "env", "time", "nohup", "nice", "command", "exec"):
            sudo |= toks[0] in ("sudo", "doas"); toks = toks[1:]
            while toks and toks[0].startswith("-"):
                toks = toks[1:]
        if toks:
            out.append(dict(prog=os.path.basename(toks[0]), args=toks[1:], raw=seg, sudo=sudo))
    return out


def overwrite_redirect(raw):
    return re.search(r"(?<![>&2\d])>(?!>)\s*(?!&)\S", raw.replace("2>", "").replace("&>", "")) is not None


def labels(cmd):
    segs = segments(cmd)
    if not segs:
        return {}
    L = {}
    progs = [s["prog"] for s in segs]
    sub = lambda s: next((a for a in s["args"] if not a.startswith("-")), "")
    def known(s):
        """programma (e sottocomando) di cui le regole conoscono gli effetti; strumenti remoti/cloud esclusi: dal
        comando non si capisce cosa cancellano, installano o pubblicano."""
        p, sc = s["prog"], sub(s)
        if p in REMOTE:
            return False
        if p == "git":
            return sc in GIT_RO | GIT_MOD | GIT_NET
        if p in ("docker", "podman"):
            return sc in DOCKER_RO | {"rm", "rmi", "stop", "start", "kill", "build", "push", "login"}
        return p in READONLY | LOCAL_MODIFY | NET_PROGS or p in PKG or p in ("systemctl", "sed", "awk", "sort", "echo")
    redirect = any(overwrite_redirect(s["raw"]) for s in segs)
    # ---- del
    d = None
    for s in segs:
        p, a = s["prog"], s["args"]
        if p in ("rm", "rmdir", "shred", "unlink", "srm", "truncate", "wipefs") or p.startswith("mkfs"):
            d = 1
        elif p == "find" and ("-delete" in a or ("-exec" in a and "rm" in a)):
            d = 1
        elif p == "git" and (sub(s) in ("clean", "rm") or (sub(s) == "reset" and "--hard" in a)):
            d = 1
        elif p == "dd" and any(x.startswith("of=") for x in a):
            d = 1
        elif p == "sed" and any(x.startswith("-i") for x in a):
            d = 1
    if d is None and redirect:
        d = 1
    if d is None and all(known(s) for s in segs):
        ambiguous = any(s["prog"] in ("cp", "mv", "install", "rsync", "tar", "unzip", "7z", "ffmpeg", "convert", "magick",
                                      "mogrify", "patch", "docker", "podman") for s in segs)
        if not ambiguous:
            d = 0
    if d is not None:
        L["del"] = d
    # ---- net
    n = None
    for s in segs:
        p, a, sc = s["prog"], s["args"], sub(s)
        if p in NET_PROGS or (p == "git" and sc in GIT_NET) or (p in PKG and sc in PKG[p] | {"update", "download"}) \
                or (p == "pacman" and any(x.startswith("-S") for x in a)) or (p in ("docker", "podman") and sc in ("pull", "push", "login", "search")) \
                or (p == "rsync" and any(":" in x for x in a)) or (s["prog"], sc) in PUSH:
            n = 1
    if n is None and URL.search(cmd):
        n = 1
    if n is None and all(s["prog"] in READONLY - NET_PROGS or s["prog"] in LOCAL_MODIFY - {"rsync"} or
                         (s["prog"] == "git" and sub(s) in GIT_RO | {"add", "commit", "merge", "rebase", "checkout", "switch",
                                                                      "reset", "restore", "stash", "tag", "init", "branch"})
                         for s in segs):
        n = 0
    if n is not None:
        L["net"] = n
    # ---- sudo (solo se il comando è nel testo: gestito da chi usa le etichette)
    root_typical = {"apt", "apt-get", "dnf", "yum", "pacman", "zypper", "mount", "umount", "fdisk", "parted", "useradd",
                    "userdel", "usermod", "groupadd", "iptables", "ufw", "modprobe", "rmmod", "insmod", "reboot", "shutdown",
                    "poweroff", "halt", "chown", "systemctl", "service", "sysctl", "swapon", "swapoff", "hwclock", "dd"}
    if any(s["sudo"] for s in segs) or re.search(r"\bsu\s+-c\b", cmd):
        L["sudo"] = 1
    elif not any(s["prog"] in root_typical or s["prog"].startswith("mkfs") for s in segs):
        L["sudo"] = 0
    # ---- inst
    i = None
    for s in segs:
        p, a, sc = s["prog"], s["args"], sub(s)
        if (p in PKG and sc in PKG[p]) or (p == "pacman" and any(re.fullmatch(r"-S[a-z]*|-U", x) for x in a) and "-Ss" not in a and "-Si" not in a) \
                or (p in ("dpkg",) and "-i" in a) or (p == "rpm" and any(x.startswith("-i") or x.startswith("-U") for x in a)) \
                or (p == "make" and "install" in a):
            i = 1
    if i is None and all(known(s) and not (s["prog"] in PKG and sub(s) not in PKG_RO) and not (s["prog"] in ("docker", "podman") and sub(s) == "pull")
                         for s in segs):
        i = 0
    if i is not None:
        L["inst"] = i
    # ---- push
    pu = 1 if any((s["prog"], sub(s)) in PUSH for s in segs) else (0 if all(known(s) for s in segs) else None)
    if pu is not None:
        L["push"] = pu
    # ---- ro
    def tar_list(s):
        return s["prog"] == "tar" and bool(s["args"]) and "t" in s["args"][0].lstrip("-") and not set("cxru") & set(s["args"][0].lstrip("-"))
    def seg_ro(s):
        p, a, sc = s["prog"], s["args"], sub(s)
        if tar_list(s):
            return True
        if p == "find":
            return not ({"-delete", "-exec", "-execdir", "-ok"} & set(a))
        if p == "sed":
            return not any(x.startswith("-i") for x in a)
        if p in ("sort",):
            return "-o" not in a
        if p in ("echo", "awk"):
            return True
        if p == "git":
            return sc in GIT_RO
        if p in ("docker", "podman"):
            return sc in DOCKER_RO
        if p == "systemctl":
            return sc in ("status", "list-units", "list-unit-files", "is-active", "is-enabled", "show", "cat")
        if p in PKG:
            return sc in PKG_RO
        return p in READONLY
    def seg_mod(s):
        p, sc = s["prog"], sub(s)
        if tar_list(s) or (p == "git" and sc == "config"):
            return False
        return (p in LOCAL_MODIFY or (p == "git" and sc in GIT_MOD) or (p in PKG and sc in PKG[p])
                or (p in ("docker", "podman") and sc in ("run", "rm", "rmi", "stop", "start", "build", "kill", "pull", "push", "exec", "create"))
                or (p == "systemctl" and sc in ("start", "stop", "restart", "enable", "disable", "reload", "mask", "unmask"))
                or (p == "sed" and any(x.startswith("-i") for x in s["args"])) or (p == "find" and {"-delete", "-exec"} & set(s["args"])))
    if not redirect and all(seg_ro(s) for s in segs):
        L["ro"] = 1
    elif redirect or any(seg_mod(s) for s in segs) or L.get("del") == 1 or L.get("inst") == 1 or L.get("push") == 1:
        L["ro"] = 0
    return L


def parse_page(path):
    """-> [(descrizione, comando)] degli esempi di una pagina tldr."""
    ex, desc = [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("- "):
            desc = line[2:].rstrip(":").strip()
        elif line.startswith("`") and line.endswith("`") and desc is not None:
            ex.append((re.sub(r"\[(\w+)\]", r"\1", desc), line[1:-1])); desc = None
    return ex


def split_of(page):
    h = int(hashlib.sha1(page.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else ("dev" if h == 1 else "train")


def download(dst):
    import tarfile, urllib.request
    os.makedirs(dst, exist_ok=True)
    tgz = os.path.join(dst, "tldr.tar.gz")
    urllib.request.urlretrieve(f"https://github.com/{TLDR[0]}/archive/{TLDR[1]}.tar.gz", tgz)
    with tarfile.open(tgz) as tf:
        members = [m for m in tf.getmembers() if "/pages" in m.name and m.name.endswith(".md")]
        for m in members:
            m.name = m.name.split("/", 1)[1]
        tf.extractall(dst, members=members)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tldr", default="data/tldr")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="bench/data/agent")
    a = ap.parse_args()
    if a.download:
        download(a.tldr)
    rng = random.Random(61)
    # i testi del test v0 (bench/build_ood_v0.py, AGENT) non devono comparire in nessuno split
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_ood_v0 import AGENT
    norm = lambda x: " ".join(x.split()).lower()
    v0 = {norm(r[0]) for r in AGENT}
    n_v0 = 0
    rows, seen_cmd = [], set()
    for plat in PLATFORMS:
        d = os.path.join(a.tldr, "pages", plat)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md"):
                continue
            page = f"{plat}/{fn[:-3]}"
            en = parse_page(os.path.join(d, fn))
            tr = {}
            for l in LANGS:
                p = os.path.join(a.tldr, f"pages.{l}", plat, fn)
                if os.path.exists(p):
                    t = parse_page(p)
                    # allineamento per comando (le traduzioni conservano i comandi)
                    tr[l] = {c: ds for ds, c in t}
            for j, (desc, cmd) in enumerate(en):
                if cmd in seen_cmd or cmd.startswith("tldr "):
                    continue
                seen_cmd.add(cmd)
                lab = labels(cmd.replace("{{", "").replace("}}", ""))
                if not lab:
                    continue
                descs = {"en": desc}
                for l in LANGS:
                    if l in tr and cmd in tr[l]:
                        descs[l] = tr[l][cmd]
                rc = render(cmd, rng)
                if norm(rc) in v0 or norm(cmd.replace("{{", "").replace("}}", "")) in v0:
                    n_v0 += 1
                    continue
                rows.append(dict(id=f"tldr-{page}-{j}", page=page, split=split_of(page), cmd=rc, desc=descs, labels=lab))
    os.makedirs(a.out, exist_ok=True)
    for sp in ("train", "dev", "test"):
        with open(os.path.join(a.out, f"tldr_{sp}.jsonl"), "w") as f:
            for r in rows:
                if r["split"] == sp:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(r["split"] for r in rows)
    pc = {p: collections.Counter(r["labels"][p] for r in rows if p in r["labels"]) for p in PROPS}
    nl = collections.Counter(len(r["desc"]) for r in rows)
    print("esclusi perché identici al test v0:", n_v0)
    print("esempi", dict(c), "\netichette", {p: dict(v) for p, v in pc.items()}, "\nlingue per esempio", dict(nl))


if __name__ == "__main__":
    main()
