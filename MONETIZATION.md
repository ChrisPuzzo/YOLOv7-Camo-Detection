# Monetization & Distribution Strategy

## TL;DR

Two SKUs, same engine:

| | Camo Detector — Beta | Camo Detector — Pro |
|---|---|---|
| **Price** | Free | Paid (one-time, $2.99–$4.99) |
| **Ads** | Yes (AdMob/AppLovin) | No |
| **Classes** | `camo` only | `camo` + future weapon classes |
| **Features** | Live detection, basic UI | Detection history, multi-class, training mode, no ads |
| **Goal** | Top-of-funnel, gather feedback, iterate on detection quality | Revenue from committed users + airsoft community |

## Why this split

### The weapons-content problem

Google explicitly bans airsoft / paintball / BB-gun *advertising*:

> "Examples of products that will no longer be allowed include paintball guns, airsoft guns, BB guns, gun scopes, ammunition belts, stun guns, and tactical knives."
> — [Google AdWords Weapons Policy summary, NLAirsoft](http://nlairsoft.com/item/google-banning-advertisement-of-airsoft-products)

That's about ads *for* airsoft products, not ads *inside* airsoft apps — but AdMob's review process is inconsistent. An app whose entire UI is bounding boxes around weapons is a ban risk. An app whose only detection target is "camouflaged person" is much safer.

Google Play and Apple have softer policies: they ban apps that **facilitate sale** of firearms, not apps that detect them. ([Firearms Policy Coalition coverage](https://www.firearmspolicy.org/google_play_s_developer_policy_updated_to_ban_firearm_sales_apps)). But "detect a weapon" is still in the gray zone reviewers can interpret strictly.

**The Beta avoids the problem entirely** by detecting only camouflage. The Pro tier reintroduces weapon detection, but as a feature behind a paywall — meaning it isn't ad-supported and won't trigger AdMob review on weapons grounds.

### Why give away the Beta?

- **Distribution**: airsoft is a niche; word-of-mouth in local clubs is more valuable than App Store search.
- **Data flywheel**: Beta users opt-in to "share detections that were wrong" → fuels v3 retraining.
- **Validation before paywall**: don't ask anyone to pay until you've shown detection works in the field.
- **Search ranking**: free apps build install base + ratings; they're a marketing channel for Pro.

## Beta — concrete plan

### Stores
- **iOS**: TestFlight (up to 10,000 testers, no App Store review) → public Play Store after stable
- **Android**: Play Store **internal testing** track first, then **closed beta** with email allowlist, then production

### Ad networks (in preference order)
1. **AppLovin MAX** — generally the most lenient on adjacent categories; mediates AdMob, Unity, Meta, etc. Best fill rate for niche apps.
2. **Unity LevelPlay (formerly IronSource)** — game-friendly, decent fill, less category-strict than AdMob.
3. **AdMob** — try last. If they ban us, we still have #1 and #2.

### Ad formats
- **Banner** at the bottom (low CPM but unobtrusive; matches the "hobby tool" feel)
- **Interstitial** on session start *only* (one ad → camera opens). Don't break flow.
- **No rewarded video** in v1 — adds complexity, niche audience won't engage with it.

### Store listing strategy (avoid weapons keywords)
- **Title**: "Camo Detector — Tactical Vision"
- **Subtitle**: "Spot camouflage in real-time"
- **Category**: Sports / Photo & Video (not "Tools" or "Utilities" — too broad)
- **Screenshots**: outdoor scenes only, no weapons, no soldiers. Forest/woodland aesthetic.
- **Description**: airsoft / paintball / nature observation framing. Don't say "weapon," "gun," or "military."
- **Keywords**: airsoft, paintball, camo, woodland, observation, tactical, outdoor

### What to track from day 1
- Install → first-detection conversion
- Avg session length
- FPS distribution by device tier
- False positive / false negative reports (in-app "wrong?" button)
- Beta → Pro conversion rate (after Pro ships)

## Pro — what to add

Concrete differentiators users will actually pay for:

- **No ads** — table-stakes
- **Weapon class detection** — what's in someone's hands
- **Detection history** — review last N detections with timestamps + thumbnails
- **Multi-class confidence tuning** — per-class threshold sliders
- **Training mode** — point at a target, get a stability/coverage score over time
- **Squad share** — local network broadcast of detections to teammates' phones (opt-in)
- **Higher-resolution model** — YOLO26-small instead of nano, slower but more accurate. Toggle.

Not all of these in v2 — pick 3 that ship cleanly.

## Pricing notes

- Apple takes 15% under Small Business Program (revenue < $1M/yr) — that's you.
- Google takes 15% on the first $1M.
- $2.99 → you net ~$2.54 per sale.
- 200 sales/month = ~$500/mo passive. Realistic for a niche but well-targeted hobby app with a free funnel.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| AdMob bans Beta app for weapons-adjacent content | Med | Lose ad revenue | Use AppLovin first; Beta is only camo, not weapons |
| Apple/Google reject Pro for weapon detection | Med | Pro launch delayed | Frame as "tactical training" not "weapon detection"; have screenshots ready that show context |
| Detection quality is poor in real airsoft conditions | High | Users uninstall | Field test before public launch; v2 retrains on real user data |
| Apple/Google decide later that airsoft = weapon | Low | App removal | Maintain a sideloadable APK + TestFlight build as escape hatch |

## Legal / compliance checklist (before public launch)

- [ ] Privacy Policy (required by both stores) — easy because everything is on-device, no data collected
- [ ] Terms of Service
- [ ] Age rating: 12+ (mild simulated combat / airsoft context)
- [ ] EULA disclaimer: "not for real-world tactical use"
- [ ] Trademark check: "Camo Detector" name availability
- [ ] LLC or sole-proprietor for tax separation if revenue exceeds ~$600/yr
