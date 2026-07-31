CREATE TABLE IF NOT EXISTS architectures (
    architecture_key VARCHAR PRIMARY KEY,
    architecture_label VARCHAR NOT NULL,
    source_commit VARCHAR NOT NULL,
    display_order BIGINT NOT NULL UNIQUE
);

INSERT INTO architectures VALUES
    ('tiny-transformer', '1-layer width-64 Transformer', 'a66155afa3596d7c43ed2813d45299d4f833d173', 1),
    ('gpt2-small', '12-layer width-768 Transformer', '655c650bfcd5a460c9f18089bb58e91ea6622acd', 2),
    ('transformer-4x256', '4-layer width-256 Transformer', '6630a4507c8d9a7bb33d7034210bd96e50136ce2', 3),
    ('transformer-2x128', '2-layer width-128 Transformer', '4ac12ae68373c55ed3dd30b827fb535688ed9276', 4),
    ('tied-transformer', 'T-step tied width-128 Transformer', '6c91e99a671d9edaba0571bfe2d52c516157f999', 5),
    ('t2mlr-cache', 'T²MLR gated-cache Transformer', '47646db668748230211074d0463b75aa2f91c133', 6),
    ('thoughtbubbles', 'Soft Thoughtbubbles Transformer', 'a408a7dd27ae4648571102948be2f098429dd8aa', 7),
    ('simple-rnn', 'Single-layer width-128 bidirectional RNN', '77788bde51bff99006d3512cfad34e3f62a3856f', 8),
    ('hierarchical-gru', 'Hierarchical T-step GRU', 'd0d326fc4aee81c9c9eb6d4b9298db3aabb2d7b7', 9),
    ('explicit-residue', 'Explicit residue-state Transformer', '628703bfd11f748a66e65d0dd88f30178b7348c4', 10),
    ('moe-64-5m', '64-expert width-98 MoE (5M)', 'a6767129f8316591f319b19d0e40c3085fb87386', 11),
    ('moe-64-50m', '64-expert width-312 MoE (50M)', '1eccfab7c309fc16c87c2ad381efc6407b84b809', 12),
    ('moe-64-5m-tied', '64-expert width-98 tied MoE (5M)', '73e3eb5e57fe667ad3bfd141a1aa4e5145df1f5c', 13),
    ('numeric-multiplicative', 'Multiplicative numeric recurrence', '394f4fc429b06488a75422c474aeb63e5c512aec', 14),
    ('joint-answer-bottleneck', 'Shallow joint-answer bottleneck', 'a644db33cbf7157343bac299437a5e19dcea997f', 15),
    ('canonical-residue', 'Canonical-state squaring recurrence', '88ca9897e9d7a0698a1dffc42c636fa7896d8f28', 16),
    ('associative-residue', 'Associative-memory residue recurrence', '6145506bcc830c906d11d0f4975988e98e4e399a', 17),
    ('digit-compositional', 'Digit-compositional residue recurrence', 'd522f3dd6f2181a4545ec98464737ebfcbc016f2', 18),
    ('modulus-specialized', 'Modulus-specialized residue recurrence', 'dcb76db4c7a3fc2ab93da1b76e16fc349e8bb5cc', 19),
    ('hybrid-numeric', 'Hybrid numeric residue recurrence', '2391edd434256912bfd48f249a0f9ddaacf34602', 20),
    ('residual-memory', 'Residual-memory residue recurrence', '5d84386a473173459e2412b5e1915835af421ef2', 21),
    ('modulus-masked', 'Modulus-masked residue recurrence', '25744ff5a5c41416dc6012ef678756730d2ae617', 22),
    ('stochastic-memory', 'Stochastic-memory residue recurrence', 'f4d84f52f800123a14237646db619e7b6fffc3bc', 23),
    ('reflection-invariant', 'Reflection-invariant residue recurrence', '2cd612c3916df3dcaf2d2ee703f1558e14508377', 24),
    ('residue-anchored', 'Residue-anchored reflection recurrence', 'def8986c5cb7f4af046ce1d1e7e67091208c29ef', 25),
    ('semigroup-jump', 'Semigroup-jump residue recurrence', 'ce388f4ca0988894d5dfc4f15946a1dc6101b44f', 26),
    ('periodic-factor', 'Periodic-factor residue recurrence', '8a640a007693e39e561a6877f67f9b2ab5ed0a5c', 27),
    ('multiperiod-orbit', 'Multiperiod orbit-memory recurrence', '783d539b309ff5eecab69cac2cc260ac677cad1d', 28),
    ('quotient-automata', 'Latent quotient-automata recurrence', '979c7397ddfa0688ff4cde3363fb7f3a3ecdeba6', 29)
ON CONFLICT (architecture_key) DO UPDATE SET
    architecture_label = excluded.architecture_label,
    source_commit = excluded.source_commit,
    display_order = excluded.display_order;

CREATE TABLE IF NOT EXISTS submissions (
    number BIGINT PRIMARY KEY,
    architecture_key VARCHAR,
    note VARCHAR NOT NULL,
    commit_hash VARCHAR NOT NULL,
    file_path VARCHAR NOT NULL,
    file_sha256 VARCHAR NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL,
    valid BOOLEAN NOT NULL,
    validated_bytes BIGINT,
    queued BOOLEAN NOT NULL,
    tier VARCHAR NOT NULL,
    dataset_id VARCHAR,
    dataset_label VARCHAR,
    attempts_left BIGINT,
    submission_id UUID,
    view_url VARCHAR,
    status VARCHAR NOT NULL,
    score_pct DOUBLE,
    max_t VARCHAR,
    ood_n_max_t VARCHAR,
    suite VARCHAR,
    run_id UUID,
    modal_call_id VARCHAR,
    exit_code INTEGER NOT NULL,
    command VARCHAR NOT NULL,
    raw_output VARCHAR NOT NULL
);

ALTER TABLE submissions
ADD COLUMN IF NOT EXISTS architecture_key VARCHAR;

UPDATE submissions
SET architecture_key = CASE
    WHEN number = 1 THEN 'tiny-transformer'
    WHEN number = 2 THEN 'gpt2-small'
    WHEN number = 3 THEN 'transformer-4x256'
    WHEN number = 4 THEN 'transformer-2x128'
    WHEN number IN (5, 6) THEN 'tied-transformer'
    WHEN number = 7 THEN 't2mlr-cache'
    WHEN number = 8 THEN 'thoughtbubbles'
    WHEN number = 9 THEN 'simple-rnn'
    WHEN number IN (10, 11) THEN 'hierarchical-gru'
    WHEN number IN (12, 13) THEN 'explicit-residue'
    ELSE architecture_key
END
WHERE architecture_key IS NULL;

CREATE OR REPLACE VIEW submission_summary AS
SELECT
    submissions.number,
    submissions.architecture_key,
    architectures.architecture_label,
    submissions.note,
    submissions.commit_hash,
    submissions.status,
    submissions.score_pct,
    submissions.tier,
    submissions.dataset_id,
    submissions.attempts_left,
    submissions.submission_id
FROM submissions
LEFT JOIN architectures USING (architecture_key)
ORDER BY number;

CREATE OR REPLACE VIEW architecture_results AS
WITH ranked AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY architecture_key, dataset_id
            ORDER BY number DESC
        ) AS dataset_rank
    FROM submissions
    WHERE dataset_id IN ('e1', 'e5')
)
SELECT
    architectures.display_order,
    architectures.architecture_key,
    architectures.architecture_label,
    architectures.source_commit,
    e1.status AS e1_status,
    e1.score_pct AS e1_score_pct,
    e1.view_url AS e1_view_url,
    e5.status AS e5_status,
    e5.score_pct AS e5_score_pct,
    e5.view_url AS e5_view_url
FROM architectures
LEFT JOIN ranked AS e1
    ON architectures.architecture_key = e1.architecture_key
    AND e1.dataset_id = 'e1'
    AND e1.dataset_rank = 1
LEFT JOIN ranked AS e5
    ON architectures.architecture_key = e5.architecture_key
    AND e5.dataset_id = 'e5'
    AND e5.dataset_rank = 1
ORDER BY architectures.display_order;
