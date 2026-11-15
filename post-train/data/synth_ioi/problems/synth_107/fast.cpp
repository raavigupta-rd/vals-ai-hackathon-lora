#include <bits/stdc++.h>
using namespace std;

struct Pt {
    long long x, y;
    Pt operator+(const Pt& o) const { return {x + o.x, y + o.y}; }
    Pt operator-(const Pt& o) const { return {x - o.x, y - o.y}; }
};

static inline __int128 cross(const Pt& a, const Pt& b) {
    return (__int128)a.x * b.y - (__int128)a.y * b.x;
}

static inline __int128 cross3(const Pt& o, const Pt& a, const Pt& b) {
    return cross(a - o, b - o);
}

static inline int half_plane(const Pt& v) {
    // Angle in [0, pi) -> 0, [pi, 2pi) -> 1
    return (v.y > 0 || (v.y == 0 && v.x > 0)) ? 0 : 1;
}

vector<Pt> convex_hull_ccw(vector<Pt>& p) {
    sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });
    p.erase(unique(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        return a.x == b.x && a.y == b.y;
    }), p.end());
    int n = (int)p.size();
    if (n <= 2) return p;
    vector<Pt> h;
    h.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        while (h.size() >= 2 && cross3(h[h.size() - 2], h.back(), p[i]) <= 0) h.pop_back();
        h.push_back(p[i]);
    }
    size_t t = h.size() + 1;
    for (int i = n - 2; i >= 0; i--) {
        while (h.size() >= t && cross3(h[h.size() - 2], h.back(), p[i]) <= 0) h.pop_back();
        h.push_back(p[i]);
    }
    h.pop_back();
    return h;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;

    Pt start{0, 0};
    vector<Pt> edges;
    edges.reserve(300000);

    for (int i = 0; i < n; i++) {
        int m;
        cin >> m;
        vector<Pt> pts(m);
        for (int j = 0; j < m; j++) cin >> pts[j].x >> pts[j].y;
        vector<Pt> h = convex_hull_ccw(pts);
        int sz = (int)h.size();
        // Rotate so lowest-then-leftmost vertex is first
        int idx = 0;
        for (int j = 1; j < sz; j++) {
            if (h[j].y < h[idx].y || (h[j].y == h[idx].y && h[j].x < h[idx].x)) idx = j;
        }
        rotate(h.begin(), h.begin() + idx, h.end());
        start = start + h[0];
        if (sz == 2) {
            edges.push_back(h[1] - h[0]);
            edges.push_back(h[0] - h[1]);
        } else if (sz >= 3) {
            for (int j = 0; j < sz; j++) {
                edges.push_back(h[(j + 1) % sz] - h[j]);
            }
        }
    }

    sort(edges.begin(), edges.end(), [](const Pt& a, const Pt& b) {
        int ha = half_plane(a), hb = half_plane(b);
        if (ha != hb) return ha < hb;
        return cross(a, b) > 0;
    });

    vector<Pt> verts;
    verts.reserve(edges.size() + 1);
    Pt cur = start;
    verts.push_back(cur);
    for (const auto& e : edges) {
        cur = cur + e;
        verts.push_back(cur);
    }

    // Extract clean strict-x upper and lower chains for branch-free binary search
    sort(verts.begin(), verts.end(), [](const Pt& a, const Pt& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });

    vector<Pt> lower, upper;
    for (const auto& p : verts) {
        if (!lower.empty() && lower.back().x == p.x) continue; // keep min y for x
        while (lower.size() >= 2 && cross3(lower[lower.size() - 2], lower.back(), p) <= 0)
            lower.pop_back();
        lower.push_back(p);
    }
    for (int i = 0; i < (int)verts.size(); i++) {
        if (i + 1 < (int)verts.size() && verts[i + 1].x == verts[i].x) continue; // keep max y for x
        const auto& p = verts[i];
        while (upper.size() >= 2 && cross3(upper[upper.size() - 2], upper.back(), p) >= 0)
            upper.pop_back();
        upper.push_back(p);
    }

    auto max_on_chain = [](const vector<Pt>& ch, long long A, long long B) -> long long {
        int lo = 0, hi = (int)ch.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            long long dx = ch[mid + 1].x - ch[mid].x;
            long long dy = ch[mid + 1].y - ch[mid].y;
            if ((__int128)A * dx + (__int128)B * dy > 0) lo = mid + 1;
            else hi = mid;
        }
        return A * ch[lo].x + B * ch[lo].y;
    };

    while (q--) {
        long long A, B;
        cin >> A >> B;
        long long ans;
        if (B >= 0) ans = max_on_chain(upper, A, B);
        else ans = max_on_chain(lower, A, B);
        // Also check opposite chain endpoints in case B == 0
        ans = max({ans, A * lower.front().x + B * lower.front().y,
                        A * lower.back().x + B * lower.back().y});
        cout << ans << '\n';
    }
    return 0;
}
