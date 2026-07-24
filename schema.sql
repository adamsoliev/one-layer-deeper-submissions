CREATE TABLE IF NOT EXISTS submissions (
    number BIGINT PRIMARY KEY,
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

CREATE OR REPLACE VIEW submission_summary AS
SELECT
    number,
    note,
    commit_hash,
    status,
    score_pct,
    tier,
    dataset_id,
    attempts_left,
    submission_id
FROM submissions
ORDER BY number;
