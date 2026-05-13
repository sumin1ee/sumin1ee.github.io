/*
 * Kill switch for the previous Chirpy service worker.
 * Unregisters itself, clears all caches, and refreshes open clients.
 * Safe to leave in place forever.
 */
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map((name) => caches.delete(name)));
      const reg = await self.registration;
      await reg.unregister();
      const clientList = await self.clients.matchAll({ type: 'window' });
      clientList.forEach((client) => client.navigate(client.url));
    })()
  );
});
