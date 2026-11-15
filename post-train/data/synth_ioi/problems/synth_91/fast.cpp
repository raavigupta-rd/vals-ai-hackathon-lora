#include <bits/stdc++.h>
using namespace std;

struct SAMNode {
    int nxt[26];
    int link = -1, len = 0;
    SAMNode() { memset(nxt, -1, sizeof(nxt)); }
};

struct SegNode {
    int ls = 0, rs = 0;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;

    vector<SAMNode> sam(1);
    sam.reserve(2 * n + 5);
    vector<int> pref_state(n + 1, 0);
    int last = 0;

    for (int i = 0; i < n; i++) {
        int c = s[i] - 'a';
        int cur = (int)sam.size();
        sam.emplace_back();
        sam[cur].len = sam[last].len + 1;
        int p = last;
        while (p != -1 && sam[p].nxt[c] == -1) {
            sam[p].nxt[c] = cur;
            p = sam[p].link;
        }
        if (p == -1) {
            sam[cur].link = 0;
        } else {
            int qn = sam[p].nxt[c];
            if (sam[p].len + 1 == sam[qn].len) {
                sam[cur].link = qn;
            } else {
                int clone = (int)sam.size();
                sam.push_back(sam[qn]);
                sam[clone].len = sam[p].len + 1;
                while (p != -1 && sam[p].nxt[c] == qn) {
                    sam[p].nxt[c] = clone;
                    p = sam[p].link;
                }
                sam[qn].link = sam[cur].link = clone;
            }
        }
        last = cur;
        pref_state[i + 1] = cur;
    }

    int V = (int)sam.size();
    // Counting sort states by len for bottom-up link-tree processing
    vector<int> cnt(n + 1, 0), ord(V);
    for (int i = 0; i < V; i++) cnt[sam[i].len]++;
    for (int i = 1; i <= n; i++) cnt[i] += cnt[i - 1];
    for (int i = V - 1; i >= 0; i--) ord[--cnt[sam[i].len]] = i;

    int LOG = __lg(V) + 2;
    vector<vector<int>> up(LOG, vector<int>(V, 0));
    for (int idx = 1; idx < V; idx++) {
        int u = ord[idx];
        up[0][u] = max(0, sam[u].link);
        for (int b = 1; b < LOG; b++) up[b][u] = up[b - 1][up[b - 1][u]];
    }

    // Persistent (non-destructive!) Segment Tree Merge so every state u retains its own endpos tree
    vector<SegNode> tr(1);
    tr.reserve(n * 40 + 5);
    auto new_seg = [&]() -> int {
        tr.push_back({});
        return (int)tr.size() - 1;
    };

    function<int(int, int, int)> make_pt = [&](int l, int r, int pos) -> int {
        int u = new_seg();
        if (l == r) return u;
        int mid = (l + r) >> 1;
        if (pos <= mid) tr[u].ls = make_pt(l, mid, pos);
        else tr[u].rs = make_pt(mid + 1, r, pos);
        return u;
    };
    function<int(int, int)> merge_seg = [&](int a, int b) -> int {
        if (!a || !b) return a ? a : b;
        int u = new_seg();
        tr[u].ls = merge_seg(tr[a].ls, tr[b].ls);
        tr[u].rs = merge_seg(tr[a].rs, tr[b].rs);
        return u;
    };
    function<bool(int, int, int, int, int)> has_in = [&](int u, int l, int r, int ql, int qr) -> bool {
        if (!u || ql > qr) return false;
        if (ql <= l && r <= qr) return true;
        int mid = (l + r) >> 1;
        if (ql <= mid && has_in(tr[u].ls, l, mid, ql, qr)) return true;
        if (qr > mid && has_in(tr[u].rs, mid + 1, r, ql, qr)) return true;
        return false;
    };

    vector<int> root(V, 0);
    for (int i = 1; i <= n; i++) {
        root[pref_state[i]] = merge_seg(root[pref_state[i]], make_pt(1, n, i));
    }
    for (int idx = V - 1; idx >= 1; idx--) {
        int u = ord[idx];
        int p = sam[u].link;
        if (p >= 0) root[p] = merge_seg(root[p], root[u]);
    }

    auto state_for_sub = [&](int a, int k) -> int {
        int u = pref_state[a + k - 1];
        for (int b = LOG - 1; b >= 0; b--) {
            int p = up[b][u];
            if (sam[p].len >= k) u = p;
        }
        return u;
    };

    while (q--) {
        int l, r, a, b;
        cin >> l >> r >> a >> b;
        int lo = 0, hi = min(r - l + 1, b - a + 1);
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            int st_u = state_for_sub(a, mid);
            if (has_in(root[st_u], 1, n, l + mid - 1, r)) lo = mid;
            else hi = mid - 1;
        }
        cout << lo << '\n';
    }
    return 0;
}
