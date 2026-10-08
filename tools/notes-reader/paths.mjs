import path from 'node:path';
import {fileURLToPath} from 'node:url';

export const here=path.dirname(fileURLToPath(import.meta.url));
export const siteRoot=path.resolve(here,'../..');
export const root=path.join(siteRoot,'notes/llm');
export const asset=path.join(root,'_reader');
// Keep the original SHA256 manifest unchanged; map its historical path keys.
export const migratedPath=name=>name.replace(/^后训练\//,'post-training/').replace(/^Infra\//,'infra/');
export const archivedPath=name=>'archive/'+migratedPath(name);
