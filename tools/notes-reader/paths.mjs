import path from 'node:path';
import {fileURLToPath} from 'node:url';

export const here=path.dirname(fileURLToPath(import.meta.url));
export const siteRoot=path.resolve(here,'../..');
export const root=path.join(siteRoot,'notes/llm');
export const asset=path.join(root,'_reader');
