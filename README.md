# Honami

The website for **Honami** — a UK consultancy working at the meeting point of
contemplative practice, complex-systems science, and the craft of building
technology.

**Live site:** https://www.honami.co.uk/

It is a single, self-contained `index.html`: no build step, no framework, no
dependencies. The hero is an animated HTML5 canvas (the word *Honami* rendered as
wind moving across a field of grain). Everything — markup, styles, and the canvas
engine — lives in that one file. Fonts are loaded from Google Fonts.

---

## Repository contents

| File | Purpose |
| --- | --- |
| `index.html` | The entire website (markup, CSS, and the canvas animation). |
| `CNAME` | Tells GitHub Pages to serve the site at `www.honami.co.uk`. |
| `robots.txt` | Welcomes search-engine crawlers and points to the sitemap. |
| `sitemap.xml` | Lists the site's one page so search engines can find it. |
| `site.webmanifest` | App metadata (name, colours, icons) for "add to home screen". |
| `og-image.png` | 1200×630 preview image shown when the link is shared (LinkedIn, Slack, WhatsApp, X, etc.). |
| `favicon.ico` | Multi-size browser-tab icon (16/32/48 px). |
| `icon-16.png`, `icon-32.png`, `icon-48.png` | PNG favicons for modern browsers. |
| `icon-180.png` | Apple touch icon (iOS home-screen). |
| `icon-192.png`, `icon-512.png` | Android / PWA icons referenced by the manifest. |
| `README.md` | This file. |

**Important:** every file must sit in the **root** of the repository (the same
folder as `index.html`), because the paths in the HTML are absolute (e.g.
`/favicon.ico`, `/og-image.png`). Don't move them into a subfolder.

---

## Deploying to GitHub Pages

1. Create a repository (e.g. `honami-site`) and upload **all** the files above to
   the root.
2. In the repo, go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**, select
   your `main` branch and the `/ (root)` folder, then **Save**.
4. The `CNAME` file already sets the custom domain to `www.honami.co.uk`. In the
   **Custom domain** box you should see it populated; if not, type it and save.
5. Tick **Enforce HTTPS** once it becomes available (it can take a little while for
   the certificate to be issued).

### DNS settings (at your domain registrar)

`www.honami.co.uk` is the canonical address. Point the `www` subdomain at GitHub
Pages with a **CNAME record**:

| Type | Name | Value |
| --- | --- | --- |
| CNAME | `www` | `<your-github-username>.github.io` |

It is also good practice to make the apex domain (`honami.co.uk`) redirect to the
`www` version. Add four **A records** on the apex pointing at GitHub's IPs:

| Type | Name | Value |
| --- | --- | --- |
| A | `@` | `185.199.108.153` |
| A | `@` | `185.199.109.153` |
| A | `@` | `185.199.110.153` |
| A | `@` | `185.199.111.153` |

GitHub will then redirect the apex to `www` automatically. (DNS changes can take
anywhere from a few minutes to a day to take effect.)

---

## What's been done for search engines (SEO)

The bulk of the SEO work lives inside the `<head>` of `index.html`:

- **Descriptive `<title>` and meta description** — the two things Google shows in a
  search result. Written to read naturally and contain the key terms (UK
  consultancy, human + AI alignment, change).
- **Canonical URL** (`https://www.honami.co.uk/`) — tells search engines this is the
  one true address, so the `www` and apex versions aren't treated as duplicates.
- **Open Graph and Twitter Card tags** — control the title, description, and preview
  image when the link is pasted into LinkedIn, Slack, WhatsApp, iMessage, X, etc.
  The preview image is `og-image.png`.
- **JSON-LD structured data** — a small machine-readable block describing the
  business (`ProfessionalService`), the person (`Alexander Sabine`, with ORCID), and
  the website. This helps search engines understand what the site *is* and can
  enable richer search results.
- **Favicons and web manifest** — the tab icon and the metadata used if someone adds
  the site to a phone home-screen.

The supporting `robots.txt` and `sitemap.xml` files invite crawlers in and tell them
which page to index.

### Steps you should do once the site is live

These can't be done from the code — they need you to be signed in to the relevant
service:

1. **Google Search Console** — go to <https://search.google.com/search-console>,
   add `https://www.honami.co.uk/` as a property, verify ownership (the DNS method is
   easiest), then submit `https://www.honami.co.uk/sitemap.xml`. This is the single
   most useful thing for getting indexed and seeing how you appear in search.
2. **Bing Webmaster Tools** — <https://www.bing.com/webmasters> — you can import
   directly from Google Search Console in one click.
3. **Test the share preview** — paste your URL into the
   [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/) and the
   [Facebook Sharing Debugger](https://developers.facebook.com/tools/debug/) to
   confirm `og-image.png` renders correctly. (These tools also force the platforms to
   refresh their cache if you ever change the image.)
4. **Validate the structured data** — paste the URL into Google's
   [Rich Results Test](https://search.google.com/test/rich-results) to confirm the
   JSON-LD is read without errors.

---

## Editing the site

Everything is in `index.html`. Search for the section you want inside the file —
the major blocks are marked with comments such as `<!-- ================= ENQUIRE
================= -->`. The colour palette is defined once at the top of the
`<style>` block in the `:root` variables.

If you change the **page copy**, no other files need updating. If you change the
**preview image**, replace `og-image.png` (keep it 1200×630) and re-run the share
debuggers above to refresh the platforms' caches.

---

*Built as a single static page so it stays fast, portable, and dependency-free.*
