"""
Task 3 - Backtracking
=====================
Problem: Knight's Tour

Find a sequence of moves for a chess knight on an n x n board so that it
visits every square exactly once (an "open" tour; does not need to
return to the start).

--------------------------------------------------------------------------
Backtracking formulation
--------------------------------------------------------------------------
State: current square + set of visited squares + move count so far.
At each step, try each of the (up to 8) legal knight moves from the
current square that lands on an unvisited square on the board. Recurse.
If a move leads to a dead end (no further valid moves and the board is
not fully covered), backtrack: unmark the square and try the next
candidate move.

--------------------------------------------------------------------------
Pruning strategy: Warnsdorff's heuristic
--------------------------------------------------------------------------
Plain backtracking explores moves in a fixed order (e.g. always trying
move offsets in the same sequence) and only prunes when it hits a dead
end -- this is "prune after the fact" and is extremely slow for boards
larger than about 5x5 because the search tree is enormous before any
dead end is detected.

Warnsdorff's heuristic prunes *before* the fact: at each step, always
move to the unvisited neighbour that itself has the FEWEST onward moves
(the most "constrained" square). Intuitively this visits corners and
edges early, before they become unreachable, drastically cutting the
branching factor in practice. Ties are broken arbitrarily (here: by
board order), which is sufficient for correctness though not guaranteed
to always succeed; on ties/failures we fall back to backtracking from
that point.

--------------------------------------------------------------------------
Complexity
--------------------------------------------------------------------------
Naive backtracking (no heuristic ordering):
    Worst case O(8^(n^2)) -- at each of the n^2 cells up to 8 choices.
    This is why naive backtracking is only feasible for very small n
    (5x5 or 6x6) within a reasonable time budget.

With Warnsdorff's heuristic as a *move-ordering* pruning strategy:
    In practice, a full tour is usually found in close to O(n^2) time
    (one decision per square, with an O(1)-O(8) neighbour-count check
    per candidate, i.e. small constant work per square) -- an enormous
    practical improvement over the exponential worst case, though
    Warnsdorff's heuristic is NOT proven to always succeed and the
    implementation below still falls back to true backtracking
    (undoing a move and trying the next-best neighbour) whenever the
    heuristic's first choice leads to a dead end, so the true worst
    case remains exponential.

Space complexity: O(n^2) for the board/visited state and O(n^2) for the
recursion stack in the worst case (one board square deep).
"""

import time


KNIGHT_MOVES = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]


def is_valid(x, y, n, board):
    return 0 <= x < n and 0 <= y < n and board[x][y] == -1


def count_onward_moves(x, y, n, board):
    """Number of unvisited squares reachable from (x, y). Used by
    Warnsdorff's heuristic to rank candidate moves."""
    count = 0
    for dx, dy in KNIGHT_MOVES:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny, n, board):
            count += 1
    return count


def knights_tour(n, start=(0, 0), use_warnsdorff=True, move_limit=None):
    """
    Attempt to find an open Knight's Tour on an n x n board.

    Args:
        n: board size
        start: (row, col) starting square
        use_warnsdorff: if True, order candidate moves by Warnsdorff's
            heuristic (fewest onward moves first) -- this is the pruning
            strategy required by the brief. If False, moves are tried in
            a fixed, unordered sequence (plain backtracking baseline),
            useful for demonstrating how much the heuristic helps.
        move_limit: optional cap on total recursive calls, used to keep
            the "no heuristic" baseline from running forever on larger
            boards during empirical comparison.

    Returns:
        (board, stats) where board[i][j] = move number (0-indexed) at
        which square (i, j) was visited, or None if no tour was found
        within the move_limit. stats is a dict with 'calls' (number of
        recursive calls made) for reporting search-space size.
    """
    board = [[-1] * n for _ in range(n)]
    board[start[0]][start[1]] = 0
    stats = {"calls": 0}

    def backtrack(x, y, move_count):
        stats["calls"] += 1
        if move_limit is not None and stats["calls"] > move_limit:
            return False
        if move_count == n * n:
            return True  # every square visited

        candidates = []
        for dx, dy in KNIGHT_MOVES:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, n, board):
                candidates.append((nx, ny))

        if use_warnsdorff:
            # Prune: try the most-constrained square first so we don't
            # accidentally strand it for later (Warnsdorff's heuristic).
            candidates.sort(key=lambda c: count_onward_moves(c[0], c[1], n, board))

        for nx, ny in candidates:
            board[nx][ny] = move_count
            if backtrack(nx, ny, move_count + 1):
                return True
            board[nx][ny] = -1  # backtrack: undo the move

        return False

    found = backtrack(start[0], start[1], 1)
    return (board if found else None), stats


def print_board(board):
    n = len(board)
    width = len(str(n * n))
    for row in board:
        print(" ".join(f"{cell:>{width}}" for cell in row))


def demo():
    print("=" * 70)
    print("Knight's Tour - Backtracking with Warnsdorff pruning demo")
    print("=" * 70)

    # Small board with Warnsdorff's heuristic
    n = 8
    print(f"\nSolving {n}x{n} Knight's Tour with Warnsdorff's heuristic...")
    start_t = time.perf_counter()
    board, stats = knights_tour(n, start=(0, 0), use_warnsdorff=True)
    elapsed = time.perf_counter() - start_t
    if board:
        print(f"Tour found in {elapsed:.4f}s using {stats['calls']} recursive calls.")
        print_board(board)
    else:
        print("No tour found.")

    # Compare search-space size: heuristic vs plain backtracking, small board
    print("\nComparing search effort: plain backtracking vs Warnsdorff's heuristic")
    print("(move_limit caps the plain search so it terminates in reasonable time)")
    for size in (5, 6):
        _, stats_plain = knights_tour(size, use_warnsdorff=False, move_limit=2_000_000)
        _, stats_heur = knights_tour(size, use_warnsdorff=True)
        print(f"  n={size}: plain backtracking calls (capped) = {stats_plain['calls']:>10,} | "
              f"Warnsdorff calls = {stats_heur['calls']:>6,}")

    # Empirical timing of the heuristic version across board sizes
    print("\nEmpirical timing of Warnsdorff-guided search for increasing board size:")
    for size in (5, 8, 10, 15, 20):
        start_t = time.perf_counter()
        board, stats = knights_tour(size, use_warnsdorff=True)
        elapsed = time.perf_counter() - start_t
        status = "found" if board else "NOT found"
        print(f"  n={size:>3} ({size*size:>4} squares): {elapsed:.4f}s, "
              f"{stats['calls']:>6,} calls, tour {status}")


if __name__ == "__main__":
    demo()
