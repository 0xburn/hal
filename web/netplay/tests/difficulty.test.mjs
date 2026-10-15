import assert from 'node:assert/strict';
import { test } from 'node:test';
import { toDifficulty, toReturn } from '../lib/roster.ts';

test('difficulty choices send exact 059 return conditioning', () => {
  assert.equal(toReturn('easy'), 20);
  assert.equal(toReturn('medium'), 120);
  assert.equal(toDifficulty(20), 'easy');
  assert.equal(toDifficulty(120), 'medium');
  assert.throws(() => toDifficulty(25), /Unsupported difficulty/);
});
