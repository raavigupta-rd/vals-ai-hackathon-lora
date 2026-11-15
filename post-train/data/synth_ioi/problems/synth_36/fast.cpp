#include <bits/stdc++.h>
using namespace std;

struct Pt {
    long long x, y;
    bool operator<(const Pt& o) const { return x != o.x ? x < o.x : y < o.y; }
    bool operator==(const Pt& o) const { return x == o.x && y == o.y; }
};

static inline __int128 cross(const Pt& a, const Pt& b, const Pt& c) {
    return (__int128)(b.x - a.x) * (c.y - a.y) - (__int128)(b.y - a.y) * (c.x - a.x);
}

static inline long long dot(const Pt& p, long long a, long long b) {
    return a * p.x + b * p.y;
}

long long max_on_chain(const vector<Pt>& h, long long a, long long b) {
    int lo = 0, hi = (int)h.size() - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (dot(h[mid], a, b) < dot(h[mid + 1], a, b)) lo = mid + 1;
        else hi = mid;
    }
    return max({dot(h.front(), a, b), dot(h.back(), a, b), dot(h[lo], a, b)});
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<Pt> pts(n);
    for (int i = 0; i < n; i++) cin >> pts[i].x >> pts[i].y;
    sort(pts.begin(), pts.end());
    pts.erase(unique(pts.begin(), pts.end()), pts.end());

    vector<Pt> low, up;
    for (const auto& p : pts) {
        while (low.size() >= 2 && cross(low[low.size() - 2], low.back(), p) <= 0) low.pop_back();
        low.push_back(p);
        while (up.size() >= 2 && cross(up[up.size() - 2], up.back(), p) >= 0) up.pop_back();
        up.push_back(p);
    }

    while (q--) {
        long long a, b;
        cin >> a >> b;
        if (b == 0) {
            cout << max(dot(pts.front(), a, b), dot(pts.back(), a, b)) << '\n';
        } else if (b > 0) {
            cout << max_on_chain(up, a, b) << '\n';
        } else {
            cout << max_on_chain(low, a, b) << '\n';
        }
    }
    return 0;
}
