import { createStore } from './jsonStore.js'

// Set of recording filenames the owner has hidden from the public demo page.
// Stored as a JSON array through the shared atomic/serialized store, so
// concurrent hide/unhide toggles can't lose an update or corrupt the file.
const store = createStore('hidden.json', { defaultValue: [] })

export async function readHidden() {
  return new Set(await store.read())
}

export async function setHidden(filename, hidden) {
  const next = await store.update(list => {
    const set = new Set(list)
    if (hidden) set.add(filename)
    else set.delete(filename)
    return [...set]
  })
  return new Set(next)
}
